# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:05:16 2017
Updated on Nov 14 2017
@author: Zain
"""
import os
import pandas as pd
import gc

import src.trainer as trainer
import src.utility as utility
import src.simple_trainer as simple
from statistics import mean

NR_FRAMES_TO_SEC = {32: 1, 94: 3, 157: 5, 188: 6, 313: 10, 628: 20, 911: 30}
TRAINING_MODE = False

if __name__ == '__main__':

    '''
    1s 32 frames
    3s 94 frames
    5s 157 frames
    6s 188 frames
    10s 313 frames
    20s 628 frames
    29.12s 911 frames
    '''

    slice_lengths = [911, 628, 313, 157, 94, 32]
    random_state_list = [0, 21, 42]
    iterations = 1
    summary_metrics_output_folder = 'trials_song_split'

    plots = True
    album_split = False

    for slice_len in slice_lengths:

        scores = []
        pooling_scores = []
        histories = []

        # GRU dropout?
        batch_size = 4
        nb_epochs = 2
        early_stop = 10  # patience
        learning_rate = 0.001


        def get_info():
            time = NR_FRAMES_TO_SEC[slice_len]
            level = 'album_split' if album_split else 'song_split'

            _stamp = f'{time}{batch_size}{nb_epochs}{early_stop}{learning_rate}{level}'
            _information = (f'iterations {iterations} |\n'
                            f'split {time}s | batch size {batch_size} | epochs {nb_epochs} |\n'
                            f'early stop {early_stop} | learning rate {learning_rate} | level {level}')
            return _information, _stamp


        if TRAINING_MODE:
            for i in range(iterations):
                score, pooling_score, history = trainer.train_model(
                    nb_classes=20,
                    slice_length=slice_len,
                    train=True,
                    load_checkpoint=True,
                    plots=plots,
                    album_split=album_split,
                    random_states=random_state_list[i],
                    save_metrics=True,
                    save_metrics_folder='metrics_song_split',
                    save_weights_folder='weights_song_split',
                    batch_size=batch_size,
                    nb_epochs=nb_epochs,
                    early_stop=early_stop,
                    lr=learning_rate)

                scores.append(score['weighted avg'])
                pooling_scores.append(pooling_score['weighted avg'])
                histories.append(history) if history and plots else None
                gc.collect()

            os.makedirs(summary_metrics_output_folder, exist_ok=True)

            pd.DataFrame(scores).to_csv(
                '{}/{}_score.csv'.format(summary_metrics_output_folder, slice_len))

            pd.DataFrame(pooling_scores).to_csv(
                '{}/{}_pooled_score.csv'.format(
                    summary_metrics_output_folder, slice_len))
        else:
            print(f'TIME: {NR_FRAMES_TO_SEC[slice_len]}S - - - - - - - - - - - - - - - - - - - - - - - -')

            for i in range(iterations):
                print(f'ITER: {i+1} - - - - - - - - -')

                f1_score, f1_pooling_score, history = simple.train_model(
                    nb_classes=20,
                    slice_length=slice_len,
                    train=True,
                    load_checkpoint=False,
                    plots=plots,
                    album_split=album_split,
                    random_states=random_state_list[i],
                    save_metrics_folder='metrics_song_split',
                    save_weights_folder='weights_song_split',
                    batch_size=batch_size,
                    nb_epochs=nb_epochs,
                    early_stop=early_stop,
                    lr=learning_rate)

                scores.append(f1_score)
                pooling_scores.append(f1_pooling_score)
                histories.append(history) if history and plots else None
                gc.collect()


            def get_mean_f1_score(_scores):
                return {
                    key: mean(report[key] for report in _scores)
                    for key in ['micro_avg', 'macro_avg', 'weighted_avg']
                }


            mean_f1_score = get_mean_f1_score(scores)
            mean_f1_pooling_scores = get_mean_f1_score(pooling_scores)

            information, stamp = get_info()
            with open(f'training _curves{os.sep}{stamp}.txt', 'w') as file:
                file.writelines([information,
                                 '\nframe-level mean f1-scores:\n' + str(mean_f1_score),
                                 '\nsong-level mean f1-scores:\n' + str(mean_f1_pooling_scores)])

        if histories:
            information, stamp = get_info()
            utility.plot_mean_history(histories, information, stamp)
