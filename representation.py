import gc
import os
from os.path import exists

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from keras.optimizers import Adam
from sklearn.manifold import TSNE

import src.models as models
import src.utility as utility
from keras import models as keras_models

if __name__ == '__main__':

    # set these parameters
    random_states = 21
    slice_length = 313
    checkpoint_path = 'weights_song_split/20_313_0'
    batch_size = 4

    # leave as-is
    load_checkpoint = True
    nb_classes = 20
    folder = 'song_data'
    lr = 0.0001  # not used
    ensemble_visual = False  # average out representations at the song level
    save_path = 'representation_output/'

    # Load the song data and split into train and test sets at song level
    print("Loading data for {}".format(slice_length))
    Y, X, S = utility.load_dataset(song_folder_name=folder,
                                   nb_classes=nb_classes,
                                   random_state=random_states)
    X, Y, S = utility.slice_songs(X, Y, S, length=slice_length)

    # Reshape data as 2d convolutional tensor shape
    X_shape = X.shape + (1,)
    X = X.reshape(X_shape)

    # encode Y
    Y_original = Y
    Y, le, enc = utility.encode_labels(Y)

    # build the model
    model = models.crnn2d(X.shape, nb_classes=Y.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])

    # Initialize weights using checkpoint if it exists
    if exists(checkpoint_path):
        print('Checkpoint file detected. Loading weights.')
        model = keras_models.load_model(checkpoint_path)
    else:
        raise Exception('no checkpoint for {}'.format(checkpoint_path))

    # drop final dense layer and activation
    print("Modifying model and predicting representation")
    model.pop()
    model.pop()
    model.summary()

    # predict representation
    print("Predicting")
    X_rep = model.predict(X, batch_size=batch_size)

    print("Garbage collection")
    del X
    gc.collect()

    if ensemble_visual:
        songs = np.unique(S)
        X_song = np.zeros((songs.shape[0], X_rep.shape[1]))
        Y_song = np.empty((songs.shape[0]), dtype="S10")
        for i, song in enumerate(songs):
            xs = X_rep[S == song]
            Y_song[i] = Y_original[S == song][0]
            X_song[i, :] = np.mean(xs, axis=0)

        X_rep = X_song
        Y_original = Y_song

    # fit tsne
    print("Fitting TSNE {}".format(X_rep.shape))
    tsne_model = TSNE()
    X_2d = tsne_model.fit_transform(X_rep)

    # save results
    print("Saving results")
    os.makedirs(save_path, exist_ok=True)
    save_path += str(checkpoint_path.split('_')[1])
    if ensemble_visual:
        save_path += '_ensemble'

    pd.DataFrame({'x0': X_2d[:, 0], 'x1': X_2d[:, 1],
                  'label': Y_original}).to_csv(
        save_path + '.csv', index=False)

    # save figure
    sns.set_palette("Paired", n_colors=20)
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1],
                    hue=Y_original, palette=sns.color_palette(n_colors=20))
    plt.savefig(save_path + '.png')

    del Y, S, X_rep, X_2d, Y_original
