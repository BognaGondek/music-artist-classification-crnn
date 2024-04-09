import src.utility as utility
import src.models as models

import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import models as keras_models

from sklearn.metrics import confusion_matrix, classification_report


def train_model(nb_classes=20,
                slice_length=911,
                artist_folder='artists',
                song_folder='song_data',
                plots=True,
                train=True,
                load_checkpoint=False,
                save_metrics=True,
                save_metrics_folder='metrics',
                save_weights_folder='weights',
                batch_size=4,
                nb_epochs=2,
                early_stop=10,
                lr=0.0001,
                album_split=True,
                random_states=42):
    """
    Main function for training the model and testing
    """

    weights = os.path.join(save_weights_folder, str(nb_classes) +
                           '_' + str(slice_length) + '_' + str(random_states))
    os.makedirs(save_weights_folder, exist_ok=True)
    os.makedirs(save_metrics_folder, exist_ok=True)

    print("Loading dataset...")

    if album_split:
        y_train, x_train, s_train, y_test, x_test, s_test, \
            y_val, x_val, s_val = \
            utility.load_dataset_album_split(song_folder_name=song_folder,
                                             artist_folder=artist_folder,
                                             nb_classes=nb_classes,
                                             random_state=random_states)
    else:
        y_train, x_train, s_train, y_test, x_test, s_test, \
            y_val, x_val, s_val = \
            utility.load_dataset_song_split(song_folder_name=song_folder,
                                            artist_folder=artist_folder,
                                            nb_classes=nb_classes,
                                            random_state=random_states)

    print("Loaded and split dataset. Slicing songs...")

    # Create slices out of the songs
    x_train, y_train, s_train = utility.slice_songs(x_train, y_train, s_train,
                                                    length=slice_length)
    x_val, y_val, s_val = utility.slice_songs(x_val, y_val, s_val,
                                              length=slice_length)
    x_test, y_test, s_test = utility.slice_songs(x_test, y_test, s_test,
                                                 length=slice_length)

    print("Training set label counts:", np.unique(y_train, return_counts=True))

    # Encode the target vectors into one-hot encoded vectors
    y_train, le, enc = utility.encode_labels(y_train)
    y_test, le, enc = utility.encode_labels(y_test, le, enc)
    y_val, le, enc = utility.encode_labels(y_val, le, enc)

    # Reshape data as 2d convolutional tensor shape
    x_train = x_train.reshape(x_train.shape + (1,))
    x_val = x_val.reshape(x_val.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    # Build the model
    model = models.crnn2d(x_train.shape, nb_classes=y_train.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr),
                  metrics=['accuracy'])
    model.summary()

    # Initialize weights using checkpoint if it exists
    if load_checkpoint:
        print("Looking for previous weights...")
        if exists(weights):
            print('Checkpoint file detected. Loading weights.')
            model = keras_models.load_model(weights)
        else:
            print('No checkpoint file detected.  Starting from scratch.')
    else:
        print('Starting from scratch (no checkpoint)')

    checkpointer = ModelCheckpoint(filepath=weights,
                                   verbose=1,
                                   save_best_only=True)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0,
                                  patience=early_stop, verbose=0, mode='auto')

    # Train the model
    history = None
    if train:
        print("Input Data Shape", x_train.shape)
        history = model.fit(x_train, y_train, batch_size=batch_size,
                            shuffle=True, epochs=nb_epochs,
                            verbose=1, validation_data=(x_val, y_val),
                            callbacks=[checkpointer, early_stopper])
        if plots:
            history = {key: value for key, value in history.history.items()}

    # Load weights that gave the best performance on validation set
    model = keras_models.load_model(weights)
    filename = os.path.join(save_metrics_folder, str(nb_classes) + '_'
                            + str(slice_length)
                            + '_' + str(random_states) + '.txt')

    # Score test model
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    y_score = model.predict(x_test, batch_size=batch_size, verbose=0)

    # Calculate confusion matrix
    y_predict = np.argmax(y_score, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_predict)

    # Plot the confusion matrix
    class_names = np.arange(nb_classes)
    class_names_original = le.inverse_transform(class_names)

    plt.figure(figsize=(14, 14))
    utility.plot_confusion_matrix(cm, classes=class_names_original,
                                  normalize=True,
                                  title='Confusion matrix with normalization')
    if save_metrics:
        plt.savefig(filename + '.png', bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(14, 14))

    # Print out metrics
    print('Test score/loss:', score[0])
    print('Test accuracy:', score[1])
    print('\nTest results on each slice:')
    scores = classification_report(y_true, y_predict,
                                   target_names=class_names_original, zero_division=np.nan)
    scores_dict = classification_report(y_true, y_predict,
                                        target_names=class_names_original,
                                        output_dict=True,
                                        zero_division=np.nan)
    print(scores)

    # Predict artist using pooling methodology
    pooling_scores, pooled_scores_dict = \
        utility.predict_artist(model, x_test, y_test, s_test,
                               le, class_names=class_names_original,
                               slices=None, verbose=False)

    # Save metrics
    if save_metrics:
        plt.savefig(filename + '_pooled.png', bbox_inches="tight")
        plt.close()
        with open(filename, 'w') as f:
            f.write("Training data shape:" + str(x_train.shape))
            f.write('\nnb_classes: ' + str(nb_classes) +
                    '\nslice_length: ' + str(slice_length))
            f.write('\nweights: ' + weights)
            f.write('\nlr: ' + str(lr))
            f.write('\nTest score/loss: ' + str(score[0]))
            f.write('\nTest accuracy: ' + str(score[1]))
            f.write('\nTest results on each slice:\n')
            f.write(str(scores))
            f.write('\n\n Scores when pooling song slices:\n')
            f.write(str(pooling_scores))

    return scores_dict, pooled_scores_dict, history
