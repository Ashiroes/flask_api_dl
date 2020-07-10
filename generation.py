# =======================================================================================================================
# GENERATION DE MUSIQUE AVEC LSTM - GENERATION A PARTIR D'UN MODELE
# =======================================================================================================================

# Desactivation de Tensorflow GPU (sinon crash... )
import os
import pickle
import time

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization, Bidirectional
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from music21 import instrument, note, stream, chord
from keras import backend as K
from music21 import converter, instrument, note, chord



# model_name = 'modeles\LSTM_MUSIQUE-03-4.8107.hdf5'

def preparationSequences(notes, pitchnames, n_vocab):
    noteVersEntier = dict((note, number) for number, note in enumerate(pitchnames))

    # Taille de la sequence à générer
    taille_sequence = 100
    entree_reseau = []
    sortie = []

    for i in range(0, len(notes) - taille_sequence, 1):
        sequence_in = notes[i:i + taille_sequence]
        sequence_out = notes[i + taille_sequence]

        entree_reseau.append([noteVersEntier[char] for char in sequence_in])
        sortie.append(noteVersEntier[sequence_out])

    n_patterns = len(entree_reseau)

    # Changement du format de l'entrée pour qu'elle corresponde à celui attendu par le réseau LSTM
    normalisationEntree = numpy.reshape(entree_reseau, (n_patterns, taille_sequence, 1))

    # Nomalisation de l'entrée
    normalisationEntree = normalisationEntree / float(n_vocab)

    return (entree_reseau, normalisationEntree)


def creationDuReseau(network_input, n_vocab, model_name):
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

    model.load_weights(model_name)

    return model


def generate_notes(modele, entree_reseau, pitchnames, n_vocab):
    # Choix d'un séquence au hasard
    print("    Selection de la sequence")
    start = numpy.random.randint(0, len(entree_reseau) - 1)
    print("        " + str(start))

    entierVersNote = dict((number, note) for number, note in enumerate(pitchnames))

    # Selection du pattern
    pattern = entree_reseau[start]
    print("        Pattern = " + str(pattern))

    prediction_output = []

    # Generation de 500 notes
    nbNotes = 100
    i = 0
    for note_index in range(nbNotes):
        print(" Generation note ou accord : " + str(i) + "/" + str(nbNotes))

        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = modele.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = entierVersNote[index]
        print("      " + str(result))
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

        i = i + 1

    return prediction_output


def sauvegardeMidiFile(prediction_output, instru):
    offset = 0
    output_notes = []

    for pattern in prediction_output:

        # Si le pattern est un accord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instru
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        # Si le pattern est une note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instru
            output_notes.append(new_note)

        # Augmentation de l'offset afin que les notes ne s'empilent pas
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    name = 'generation/output' + str(time.time()) + '.mid'
    midi_stream.write('midi', fp=name)

    s = converter.parse(name)
    for p in s.parts:
        p.insert(0, instru)
    midi_stream = stream.Stream(s)
    midi_stream.write('midi', fp=name)

    return (name)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    APPLICATION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def generate_song(name: str, model_name: str, instru=instrument.AcousticGuitar()) -> str:
    with open('dataset/' + name + '/notes.txt', 'rb') as fichier:
        notes = pickle.load(fichier)

    frequencesNotes = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    entree_reseau, entree_normalisee = preparationSequences(notes, frequencesNotes, n_vocab)

    modele = creationDuReseau(entree_normalisee, n_vocab, model_name)

    prediction = generate_notes(modele, entree_reseau, frequencesNotes, n_vocab)

    K.clear_session()
    return sauvegardeMidiFile(prediction, instru)
