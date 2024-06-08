import src.utility as utility
import src.models as models

import os
import numpy as np
from os.path import exists

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import models as keras_models

from sklearn.metrics import classification_report

from src.attention_model import EncoderLayer, MultiHeadAttention, PositionWiseFeedForward, ScaledDotProductAttention


def train_model(nb_classes=20,
                slice_length=911,
                artist_folder='artists',
                song_folder='song_data',
                plots=True,
                train=True,
                load_checkpoint=False,
                save_metrics_folder='metrics',
                save_weights_folder='weights',
                batch_size=4,
                nb_epochs=2,
                early_stop=10,
                lr=0.0001,
                album_split=True,
                random_states=42):
    """
    Main function for training the model and validating.
    """

    weights = os.path.join(save_weights_folder, str(nb_classes) +
                           '_' + str(slice_length) + '_' + str(random_states))
    os.makedirs(save_weights_folder, exist_ok=True)
    os.makedirs(save_metrics_folder, exist_ok=True)

    # print("Loading dataset...")

    if album_split:
        y_train, x_train, s_train, _, _, _, \
            y_val, x_val, s_val = \
            utility.load_dataset_album_split(song_folder_name=song_folder,
                                             artist_folder=artist_folder,
                                             nb_classes=nb_classes,
                                             random_state=random_states)
    else:
        y_train, x_train, s_train, _, _, _, \
            y_val, x_val, s_val = \
            utility.load_dataset_song_split(song_folder_name=song_folder,
                                            artist_folder=artist_folder,
                                            nb_classes=nb_classes,
                                            random_state=random_states)

    # print("Loaded and split dataset. Slicing songs...")

    # Create slices out of the songs
    x_train, y_train, s_train = utility.slice_songs(x_train, y_train, s_train,
                                                    length=slice_length)
    x_val, y_val, s_val = utility.slice_songs(x_val, y_val, s_val,
                                              length=slice_length)

    # print("Training set label counts:", np.unique(y_train, return_counts=True))

    # Encode the target vectors into one-hot encoded vectors
    y_train, le, enc = utility.encode_labels(y_train)
    y_val, le, enc = utility.encode_labels(y_val, le, enc)

    # Reshape data as 2d convolutional tensor shape
    x_train = x_train.reshape(x_train.shape + (1,))
    x_val = x_val.reshape(x_val.shape + (1,))

    # Initialize weights using checkpoint if it exists
    if load_checkpoint and exists(weights):
        model = keras_models.load_model(weights)
    else:
        # Build the model
        model = models.crnn2d(x_train.shape, nb_classes=y_train.shape[1])
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=lr),
                      metrics=['accuracy'])
        # model.summary()

    checkpointer = ModelCheckpoint(filepath=weights,
                                   verbose=1,
                                   save_best_only=True)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0,
                                  patience=early_stop, verbose=0, mode='auto')

    # Train the model
    train_history = None
    if train:
        # print("Input Data Shape", x_train.shape)
        history = model.fit(x_train, y_train, batch_size=batch_size,
                            shuffle=True, epochs=nb_epochs,
                            verbose=1, validation_data=(x_val, y_val),
                            callbacks=[checkpointer, early_stopper])
        if plots:
            train_history = {key: value for key, value in history.history.items()}

    # Load weights that gave the best performance on validation set
    model = keras_models.load_model(weights, custom_objects={'EncoderLayer': EncoderLayer,
                                                             'PositionWiseFeedForward': PositionWiseFeedForward,
                                                             'MultiHeadAttention': MultiHeadAttention,
                                                             'ScaledDotProductAttention': ScaledDotProductAttention})

    # Score test model
    y_score = model.predict(x_val, batch_size=batch_size, verbose=0)

    # Calculate confusion matrix
    y_predict = np.argmax(y_score, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Plot the confusion matrix
    class_names = np.arange(nb_classes)
    class_names_original = le.inverse_transform(class_names)

    scores_dict = classification_report(y_true, y_predict,
                                        target_names=class_names_original,
                                        output_dict=True,
                                        zero_division=np.nan)

    # Predict artist using pooling methodology
    pooling_scores, pooled_scores_dict = \
        utility.predict_artist(model, x_val, y_val, s_val,
                               le, class_names=class_names_original,
                               slices=None, verbose=False)

    def get_f1_scores(report):
        return {'micro_avg': report['accuracy'],
                'macro_avg': report['macro avg']['f1-score'],
                'weighted_avg': report['weighted avg']['f1-score']}

    f1_scores = get_f1_scores(scores_dict)
    f1_pooling_scores = get_f1_scores(pooled_scores_dict)

    # print('f1_scores', f1_scores)
    # print('f1_pooling_scores', f1_pooling_scores)

    return f1_scores, f1_pooling_scores, train_history
