import os
import dill
import random
import itertools

import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

import librosa
import librosa.display
import librosa.feature

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from scipy import stats


def visualize_spectrogram(path, duration=None,
                          offset=0, sr=16000, n_mel=128, n_fft=2048,
                          hop_length=512):
    """This function creates a visualization of a spectrogram
    given the path to an audio file."""

    # Make a mel-scaled power (energy-squared) spectrogram
    y, sr = librosa.load(path, sr=sr, duration=duration, offset=offset)
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mel, n_fft=n_fft,
                                       hop_length=hop_length)

    # Convert to log scale (dB)
    log_s = librosa.power_to_db(s, ref=1.0)

    # Render output spectrogram in the console
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(log_s, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()


def create_dataset(artist_folder='artists', save_folder='song_data',
                   sr=16000, n_mel=128,
                   n_fft=2048, hop_length=512):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    artists = [path for path in os.listdir(artist_folder) if
               os.path.isdir(artist_folder + os.sep + path)]

    # iterate through all artists, albums, songs and find mel spectrogram
    for artist in artists:
        print(artist)
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)

        for album in artist_albums:
            album_path = os.path.join(artist_path, album)
            album_songs = os.listdir(album_path)

            for song in album_songs:
                song_path = os.path.join(album_path, song)

                # Create mel spectrogram and convert it to the log scale
                y, sr = librosa.load(song_path, sr=sr)
                s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mel,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length)
                log_s = librosa.power_to_db(s, ref=1.0)
                data = (artist, log_s, song)

                # Save each song
                save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
                with open(os.path.join(save_folder, save_name), 'wb') as fp:
                    dill.dump(data, fp)


def load_dataset(song_folder_name='song_data',
                 artist_folder='artists',
                 nb_classes=20, random_state=42):
    """This function loads the dataset based on a location;
     it returns a list of spectrograms
     and their corresponding artists/song names"""

    # Get all songs saved as numpy arrays in the given folder
    song_list = os.listdir(song_folder_name)

    # Load the list of artists
    artist_list = [path for path in os.listdir(artist_folder) if
                   os.path.isdir(artist_folder + os.sep + path)]

    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    artist = []
    spectrogram = []
    song_name = []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
        if loaded_song[0] in artists:
            artist.append(loaded_song[0])
            spectrogram.append(loaded_song[1])
            song_name.append(loaded_song[2])

    return artist, spectrogram, song_name


def load_dataset_album_split(song_folder_name='song_data',
                             artist_folder='artists',
                             nb_classes=20, random_state=42):
    """ This function loads a dataset and splits it on an album level"""
    song_list = os.listdir(song_folder_name)

    # Load the list of artists
    artist_list = [path for path in os.listdir(artist_folder) if
                   os.path.isdir(artist_folder + os.sep + path)]

    train_albums = []
    test_albums = []
    val_albums = []
    random.seed(random_state)
    for artist in os.listdir(artist_folder):
        albums = os.listdir(os.path.join(artist_folder, artist))
        random.shuffle(albums)
        test_albums.append(artist + '_%%-%%_' + albums.pop(0))
        val_albums.append(artist + '_%%-%%_' + albums.pop(0))
        train_albums.extend([artist + '_%%-%%_' + album for album in albums])

    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    y_train, y_test, y_val = [], [], []
    x_train, x_test, x_val = [], [], []
    s_train, s_test, s_val = [], [], []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
        artist, album, song_name = song.split('_%%-%%_')
        artist_album = artist + '_%%-%%_' + album

        if loaded_song[0] in artists:
            if artist_album in train_albums:
                y_train.append(loaded_song[0])
                x_train.append(loaded_song[1])
                s_train.append(loaded_song[2])
            elif artist_album in test_albums:
                y_test.append(loaded_song[0])
                x_test.append(loaded_song[1])
                s_test.append(loaded_song[2])
            elif artist_album in val_albums:
                y_val.append(loaded_song[0])
                x_val.append(loaded_song[1])
                s_val.append(loaded_song[2])

    return y_train, x_train, s_train, \
        y_test, x_test, s_test, \
        y_val, x_val, s_val


def load_dataset_song_split(song_folder_name='song_data',
                            artist_folder='artists',
                            nb_classes=20,
                            test_split_size=0.1,
                            validation_split_size=0.1,
                            random_state=42):
    y, x, s = load_dataset(song_folder_name=song_folder_name,
                           artist_folder=artist_folder,
                           nb_classes=nb_classes,
                           random_state=random_state)
    # train and test split
    x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
        x, y, s, test_size=test_split_size, stratify=y,
        random_state=random_state)

    # Create a validation to be used to track progress
    x_train, x_val, y_train, y_val, s_train, s_val = train_test_split(
        x_train, y_train, s_train, test_size=validation_split_size,
        shuffle=True, stratify=y_train, random_state=random_state)

    return y_train, x_train, s_train, \
        y_test, x_test, s_test, \
        y_val, x_val, s_val


def slice_songs(x, y, s, length=911):
    """Slices the spectrogram into sub-spectrograms according to length"""

    # Create empty lists for train and test sets
    artist = []
    spectrogram = []
    song_name = []

    # Slice up songs using the length specified
    for i, song in enumerate(x):
        slices = int(song.shape[1] / length)
        for j in range(slices - 1):
            spectrogram.append(song[:, length * j:length * (j + 1)])
            artist.append(y[i])
            song_name.append(s[i])

    return np.array(spectrogram), np.array(artist), np.array(song_name)


def create_spectrogram_plots(artist_folder='artists', sr=16000, n_mel=128,
                             n_fft=2048, hop_length=512):
    """Create a spectrogram from a randomly selected song
     for each artist and plot"""

    # get list of all artists
    artists = [path for path in os.listdir(artist_folder) if
               os.path.isdir(artist_folder + os.sep + path)]

    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(14, 12), sharex=True,
                           sharey=True)

    row = 0
    col = 0

    # iterate through artists, randomly select an album,
    # randomly select a song, and plot a spectrogram on a grid
    for artist in artists:
        print(artist)
        # Randomly select album and song
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)
        album = random.choice(artist_albums)
        album_path = os.path.join(artist_path, album)
        album_songs = os.listdir(album_path)
        song = random.choice(album_songs)
        song_path = os.path.join(album_path, song)

        # Create mel spectrogram
        y, sr = librosa.load(song_path, sr=sr, offset=60, duration=3)
        s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mel,
                                           n_fft=n_fft, hop_length=hop_length)
        log_s = librosa.power_to_db(s, ref=1.0)

        # Plot on grid
        plt.axes(ax[row, col])
        librosa.display.specshow(log_s, sr=sr)
        plt.title(artist)
        col += 1
        if col == 5:
            row += 1
            col = 0

    fig.tight_layout()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.colormaps['Blues']):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def combine_history(histories):
    """
        This function makes mean of each training history value
        (training / validation accuracy / loss) from each iteration.
        """
    mean_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    for key in mean_history.keys():
        for history in histories:
            mean_history[key].append(history[key])
        mean_history[key] = np.mean(mean_history[key], axis=0)

    return mean_history


def plot_mean_history(histories,
                      information,
                      stamp,
                      silent=True):
    """
    This function plots the training and validation accuracy
    and loss per epoch of training (averaged from each iteration).
    """
    history = combine_history(histories)
    epochs = np.arange(start=0, stop=len(histories[0]['accuracy']), step=1)

    plt.figure()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(f'Mean accuracy: {information}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(f'training _curves{os.sep}acc_{stamp}.png')
    if not silent:
        plt.show()

    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'Mean loss: {information}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(f'training _curves{os.sep}loss_{stamp}.png')
    if not silent:
        plt.show()

    return


def predict_artist(model, x, y, s,
                   le, class_names,
                   slices=None, verbose=False,
                   ml_mode=False):
    """
    This function takes slices of songs and predicts their output.
    For each song, it votes on the most frequent artist.
    """
    print("Test results when pooling slices by song and voting:")
    # Obtain the list of songs
    songs = np.unique(s)

    prediction_list = []
    actual_list = []

    # Iterate through each song
    for song in songs:

        # Grab all slices related to a particular song
        x_song = x[s == song]
        y_song = y[s == song]

        # If not using full song, shuffle and take up to a number of slices
        if slices and slices <= x_song.shape[0]:
            x_song, y_song = shuffle(x_song, y_song)
            x_song = x_song[:slices]
            y_song = y_song[:slices]

        # Get probabilities of each class
        predictions = model.predict(x_song, verbose=0)

        if not ml_mode:
            # Get list of highest probability classes and their probability
            class_prediction = np.argmax(predictions, axis=1)
            class_probability = np.max(predictions, axis=1)

            # keep only predictions confident about;
            prediction_summary_trim = class_prediction[class_probability > 0.5]

            # deal with edge case where there is no confident class
            if len(prediction_summary_trim) == 0:
                prediction_summary_trim = class_prediction
        else:
            prediction_summary_trim = predictions

        # get most frequent class
        prediction = stats.mode(prediction_summary_trim)[0]
        actual = stats.mode(np.argmax(y_song))[0]

        # Keeping track of overall song classification accuracy
        prediction_list.append(prediction)
        actual_list.append(actual)

        # Print out prediction
        if verbose:
            print(song)
            print("Predicted:", le.inverse_transform(prediction), "\nActual:",
                  le.inverse_transform(actual))
            print('\n')

    # Print overall song accuracy
    actual_array = np.array(actual_list)
    prediction_array = np.array(prediction_list)
    cm = confusion_matrix(actual_array, prediction_array)
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                          title='Confusion matrix for pooled results' +
                                ' with normalization')
    class_report = classification_report(actual_array, prediction_array,
                                         target_names=class_names,
                                         zero_division=np.nan)
    print(class_report)

    class_report_dict = classification_report(actual_array, prediction_array,
                                              target_names=class_names,
                                              output_dict=True,
                                              zero_division=np.nan)
    return class_report, class_report_dict


def encode_labels(y, le=None, enc=None):
    """Encodes target variables into numbers and then one hot encodings"""

    # initialize encoders
    n = y.shape[0]

    # Encode the labels
    if le is None:
        le = preprocessing.LabelEncoder()
        y_le = le.fit_transform(y).reshape(n, 1)
    else:
        y_le = le.transform(y).reshape(n, 1)

    # convert into one hot encoding
    if enc is None:
        enc = preprocessing.OneHotEncoder()
        y_enc = enc.fit_transform(y_le).toarray()
    else:
        y_enc = enc.transform(y_le).toarray()

    # return encoders to re-use on other data
    return y_enc, le, enc


def simple_encoding(y, le=None):
    """Encodes target variables into numbers"""

    # initialize encoders
    # N = y.shape[0]

    # Encode the labels
    if le is None:
        le = preprocessing.LabelEncoder()
        y_le = le.fit_transform(y)
    else:
        y_le = le.transform(y)

    # return encoders to re-use on other data
    return y_le, le


if __name__ == '__main__':

    # configuration options
    create_data = False
    create_visuals = True
    save_visuals = True

    if create_data:
        create_dataset(artist_folder='artists', save_folder='song_data',
                       sr=16000, n_mel=128, n_fft=2048,
                       hop_length=512)

    if create_visuals:
        # Create spectrogram for a specific song
        visualize_spectrogram(
            'artists/u2/The_Joshua_Tree/' +
            '02-I_Still_Haven_t_Found_What_I_m_Looking_For.mp3',
            offset=60, duration=29.12)

        # Create spectrogram subplots
        create_spectrogram_plots(artist_folder='artists', sr=16000, n_mel=128,
                                 n_fft=2048, hop_length=512)
        if save_visuals:
            plt.savefig(os.path.join('spectrograms.png'),
                        bbox_inches="tight")
