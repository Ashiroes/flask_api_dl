# =======================================================================================================================
# GENERATION DE MUSIQUE AVEC LSTM - APPRENTISSAGE
# =======================================================================================================================


import glob
import pickle
import time

import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils
from music21 import converter, instrument, note, chord

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    FONCTIONS UTILES
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
name = "Rock"
filepath = "modeles/" + name + "LSTM_MUSIQUE-{epoch:02d}-{loss:.4f}.hdf5"  # nom du hdf5
midifiles = "../magenta/dataset/JAZZ/MIDIFILES/piano/"  # dossier avec midifiles
note_file = "dataset/' + name + '/notes.txt"  # fichier note (genere dans ce fichier)


def extractionNotesEtAccords(chemin_fichiers_midi):
    notesEtAccords = []

    for fichier in glob.glob(chemin_fichiers_midi + "*.mid"):

        midiFile = converter.parse(fichier)

        print("Extraction des notes et accords du fichier %s" % fichier)

        notesAExtraires = None

        # Dans le cas où le fichier contient des instruments
        try:
            s2 = instrument.partitionByInstrument(midiFile)
            # On ne selectionne que le premier instrument (monophonique)
            # Il est aussi possible de tester si l'instrument est un piano... a modifier en fonction des cas d'usages
            notesAExtraires = s2.parts[0].recurse()

        # Sinon c'est un fichier plat
        except:
            notesAExtraires = midiFile.flat.notes

        for element in notesAExtraires:

            # Si c'est une note, on extrait son "Pitch"
            if isinstance(element, note.Note):
                notesEtAccords.append(str(element.pitch))

            # Si c'est un accord, on extrait chaque note
            elif isinstance(element, chord.Chord):
                notesEtAccords.append('.'.join(str(n) for n in element.normalOrder))

    # Sauvegarde des notes
    print("Nombre de notes et accords : " + str(len(notesEtAccords)))

    return notesEtAccords


def preparationSequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    tailleSequence = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # Transformation des notes en entier
    noteVersEntier = dict((note, number) for number, note in enumerate(pitchnames))

    entree_reseau = []
    sortie_reseau = []

    # create input sequences and the corresponding outputs
    # i est compris entre 0 et le nombre de notes total et la tailleDeSequence
    for i in range(0, len(notes) - tailleSequence, 1):
        # On prend l'indice i on selectionne les notes à partir de i jusqu'à la taille de séquence
        # Cela crée des séquences différentes à chaque itération
        sequence_in = notes[i:i + tailleSequence]
        sequence_out = notes[i + tailleSequence]

        # Creation de la séquence d'entrée
        entree_reseau.append([noteVersEntier[char] for char in sequence_in])

        # Creation de la séquence de sortie
        sortie_reseau.append(noteVersEntier[sequence_out])

    n_patterns = len(entree_reseau)

    print("Nombre de pattern : " + str(n_patterns))

    # On redimentionne l'entrée pour qu'elle soit compatible avec le réseau LSTM
    entree_reseau = numpy.reshape(entree_reseau, (n_patterns, tailleSequence, 1))
    print("Dimension en entrée du réseau LSTM : " + str(entree_reseau.shape))

    # On normalise les données d'entrée
    entree_reseau = entree_reseau / float(n_vocab)

    # One Hot Encoding pour la classification en sortie
    print("One hot encoding")
    print("Avant : " + str(sortie_reseau[0]))
    sortie_reseau = np_utils.to_categorical(sortie_reseau)
    print("Après : " + str(sortie_reseau[0]))

    return (entree_reseau, sortie_reseau)


def creationDuReseau(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def realisationApprentissage(modele, entree_reseau, sortie_reseau, nbEpochs, batchSize):
    # Fonction permettant de sauvegarder le meilleurs apprentissage
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    modele.fit(entree_reseau, sortie_reseau, epochs=nbEpochs, batch_size=batchSize, callbacks=callbacks_list, verbose=1)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    APPLICATION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

notes = extractionNotesEtAccords(midifiles)

pickle.dump(notes, open(note_file, "wb"))

n_vocab = len(set(notes))

entree_reseau, sortie_reseau = preparationSequences(notes, n_vocab)

modele = creationDuReseau(entree_reseau, n_vocab)

DEBUT = time.time()

realisationApprentissage(modele, entree_reseau, sortie_reseau, 100, 128)

FIN = time.time()
heures, rem = divmod(FIN - DEBUT, 3600)
minutes, secondes = divmod(rem, 60)
print("Durée d'execution :   {:0>2}:{:0>2}:{:05.2f}".format(int(heures), int(minutes), secondes))
