# =======================================================================================================================
# TRANSFERT DE STYLE
# =======================================================================================================================

import time

import IPython.display as display
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    FONCTIONS UTILES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def chargement_image(chemin_image):
    max_dim = 512

    # Chargement de l'image
    image = tf.io.read_file(chemin_image)

    # Recherche du type d'image (jpg, png...) et transformation en Tensor
    image = tf.image.decode_image(image, channels=3)
    print("        Dimensions originales de l'image (w, h, canaux) : " + str(image.shape))

    # Convertion de chaque pixel en type décimal
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Transformation de l'image en espace vectoriel (tensor) ayant une longueur de 512 pixels
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    print("        Vecteurs Tensorflow de l'image (nbImages, w , h, canaux) : " + str(image.shape))

    return image


def conversion_tenseur_vers_image(tenseur):
    tenseur = tenseur * 255

    # Transformation des valeur du tenseur en tableau d'entier de 8 bytes
    tenseur = np.array(tenseur, dtype=np.uint8)

    if (np.ndim(tenseur) > 3):
        assert tenseur.shape[0] == 1
        tenseur = tenseur[0]

    # utilisation de la librairie Pillow pour transformer le tableau en image
    return PIL.Image.fromarray(tenseur)


def affichage_image(image, titre=None):
    # Si l'image comporte plus de 3 dimensions, on supprime l'axe 0
    if (len(image.shape) > 3):
        image = tf.squeeze(image, axis=0)

    # Affichage avec matplotlib
    plt.imshow(image)
    if titre:
        plt.title(titre)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    FONCTIONS POUR LA GENERATION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def creation_nouveau_modele(modele, couches_selectionnees):
    modele.trainable = False
    couches = [modele.get_layer(name).output for name in couches_selectionnees]
    # On crée un modele avec pour entrée, le format  à VGG (image de 224 x 224)
    # Mais ne contenant que les couches selectionnées
    modele = tf.keras.Model([modele.input], couches)
    return modele


def matrice_de_Gram(tenseur):
    resultat = tf.linalg.einsum('bijc,bijd->bcd', tenseur, tenseur)
    format = tf.shape(tenseur)
    nombre_positions = tf.cast(format[1] * format[2], tf.float32)
    return resultat / (nombre_positions)


class Extracteur_Style_Contenu(tf.keras.models.Model):
    def __init__(self, modele, couches_style, couches_contenu):
        super(Extracteur_Style_Contenu, self).__init__()

        # On crée un modele VGG comportant les couches de styles et les couches de contenu
        self.vgg = creation_nouveau_modele(modele, couches_style + couches_contenu)

        self.couches_styles = couches_style
        self.couches_contenu = couches_contenu
        self.nb_couches_style = len(couches_style)

        # On n'accepte pas la mise à jour des poids lors de la rétropropagation
        self.vgg.trainable = False

    # Extraction des valeurs des couches sur une image (input)
    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0

        # Les images sont converties de RVB en BGR, puis chaque canal de couleur est centré sur zéro par rapport à l'ensemble de données ImageNet.
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        # On utilise le réseau VGG avec les couches paramétrées pour en calculer leur valeurs
        valeurs_couches = self.vgg(preprocessed_input)

        # On dispatch les valeurs des couches.
        valeurs_couches_style, valeurs_couches_contenu = (
        valeurs_couches[:self.nb_couches_style], valeurs_couches[self.nb_couches_style:])

        # Calcul de la matrice de Gram pour chaque couche de style. Cette matrice vient remplacer les valeurs des couches
        # de sorties du style
        valeurs_couches_style = [matrice_de_Gram(valeur_couche_style)
                                 for valeur_couche_style in valeurs_couches_style]

        contenu_dictionnaire = {content_name: value
                                for content_name, value
                                in zip(self.couches_contenu, valeurs_couches_contenu)}

        style_dictionnaire = {style_name: value
                              for style_name, value
                              in zip(self.couches_styles, valeurs_couches_style)}

        return {'contenu': contenu_dictionnaire, 'style': style_dictionnaire}


def calcul_du_cout(valeurs_couches, cible_contenu, cible_style, poids_style, nb_couches_style, poids_contenu,
                   nb_couches_contenu):
    valeurs_couches_style = valeurs_couches['style']
    valeurs_couches_contenu = valeurs_couches['contenu']

    # -- COUT SUR LE STYLE
    # Erreur (La génération par rapport à la cible) = MSE
    fonction_cout_style = tf.add_n([tf.reduce_mean((valeurs_couches_style[name] - cible_style[name]) ** 2)
                                    for name in valeurs_couches_style.keys()])

    # Utilisation d'un poids
    fonction_cout_style *= poids_style / nb_couches_style

    # -- COUT SUR LE CONTENU
    # Erreur sur le contenu (La génération par rapport à la cible) = MSE
    fonction_cout_contenu = tf.add_n([tf.reduce_mean((valeurs_couches_contenu[name] - cible_contenu[name]) ** 2)
                                      for name in valeurs_couches_contenu.keys()])

    fonction_cout_contenu *= poids_contenu / nb_couches_contenu

    cout = fonction_cout_style + fonction_cout_contenu
    return cout


@tf.function()
def etape_generation(image, optimiseur, extracteur, cible_contenu, cible_style, poids_style, nb_couches_style,
                     poids_contenu, nb_couches_contenu, poids_filtres_hf):
    # Creation d'un pipeline d'execution
    with tf.GradientTape() as pipeline:
        # Calcul des valeurs des couches de contenu et de style
        valeurs_couches = extracteur(image)

        # Calcul du cout total
        cout = calcul_du_cout(valeurs_couches, cible_contenu, cible_style, poids_style, nb_couches_style, poids_contenu,
                              nb_couches_contenu)

        # Reduction des hautes frequences de l'image (diminution des contours)
        cout += poids_filtres_hf * tf.image.total_variation(image)

    # Calcul du gradient
    grad = pipeline.gradient(cout, image)
    optimiseur.apply_gradients([(grad, image)])

    # Conversion des valeur de l'image entre 0 et 1
    # On remplace l'image reçue en parametre...
    image.assign(clip_0_1(image))

    return image

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    APPLICATION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def style_transfer(chemin_image_source, chemin_image_style) -> str:
    chemin_image_composite = "generation-image-"+str(time.time())

    image_source = chargement_image(chemin_image_source)
    image_style = chargement_image(chemin_image_style)

    couche_VGG19_contenu = ['block5_conv2']
    nombre_couches_contenu = len(couche_VGG19_contenu)

    couche_VGG19_style = ['block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2', 'block5_conv2']
    nombre_couches_style = len(couche_VGG19_style)

    optimiseur = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    poids_du_style_beta = 1e-2
    poids_du_contenu_alpha = 1e4

    # suppression des hautes frequences
    poids_filtres_hf = 30

    # chargement du modele vgg10 sans sa tete
    vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    # definition des cibles a atteindre
    extracteur = Extracteur_Style_Contenu(modele=vgg19, couches_contenu=couche_VGG19_contenu,
                                          couches_style=couche_VGG19_style)

    cible_style = extracteur(image_style)['style']
    cible_contenu = extracteur(image_source)['contenu']

    image = tf.Variable(image_source)

    epochs = 10
    etapes_generation_epoch = 100
    # Generarion de l'image
    DEBUT = time.time()
    step = 0

    for n in range(epochs):
        print("Epoch :" + str(n))
        for n in range(etapes_generation_epoch):
            step += 1
            etape_generation(image, optimiseur, extracteur, cible_contenu, cible_style, poids_du_style_beta,
                             nombre_couches_style, poids_du_contenu_alpha, nombre_couches_contenu, poids_filtres_hf)
            print(".", end="")
        display.clear_output(wait=True)

    FIN = time.time()

    conversion_tenseur_vers_image(image).save(chemin_image_composite)
    return chemin_image_composite
