# -*- coding: utf-8 -*-
# import os
# import click
# import logging
# from dotenv import find_dotenv, load_dotenv
#
#
# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')
#
#
# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)
#
#     # not used in this stub but often useful for finding various files
#     project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
#
#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())
#
#     main()

from scipy.io import loadmat
import numpy as np

def read_BCIIII_p300_mat(file_path):
    data = loadmat(file_path)
    del data['__globals__'],data['__header__'],data['__version__']
    return data

def find_timepoints_1D(single_stimulus_code):
    """
    Find the indexes where the value of single_stimulus_code turn from zero to non_zero
    single_stimulus_code : 1-D array

    >>> find_timepoints_1D([5,5,0,0,4,4,4,0,0,1,0,2,0])
    array([ 0,  4,  9, 11])

    """
    flag = True                            # whether have seen 0 so far
    timepoints = []
    for index,timepoint in enumerate(single_stimulus_code):
        if timepoint != 0 and flag:
            timepoints.append(index)
            flag = False
        if timepoint == 0 and not flag:
            flag = True
    return np.array(timepoints)

def find_timepoints_2D(stimulus_code):
    timepoints_two_D = []
    for row in range(stimulus_code.shape[0]):
        timepoints_two_D.append(find_timepoints_1D(stimulus_code[row]))
    return np.array(timepoints_two_D)

def timepoints_slice_second_1D (signal, timepoints, seconds_to_slice, Fs = 240):
    datas = []
    points_to_slice = int(Fs*seconds_to_slice)
    for timepoint in timepoints:
        datas.append(signal[timepoint:timepoint+points_to_slice])
    return np.array(datas)    # (len(timepoints)12*15,points_to_slice)

def timepoints_slice_second_2D(signals, timepoints_two_D, seconds_to_slice, Fs = 240):
    datas = []
    for row in range(signals.shape[0]):
        datas.append(timepoints_slice_second_1D(signals[row],timepoints_two_D[row],seconds_to_slice,Fs))
    return np.array(datas)  # (rows,len(timepoints),points_to_slice)

def timepoints_slice_second_3D(electrode_axis,signals, timepoints_two_D, seconds_to_slice, Fs = 240):
    """

    """


def prepare_data(data,seconds_to_slice,Fs=240):
    num_characters = data['Signal'].shape[0]
    num_electrodes = 64
    character_data = []
    for character in range(num_characters):
       timepoints = find_timepoints_1D(data['StimulusCode'])  # (12*15,)
       timepoints_reshaped = timepoints.reshape([12,15])









    #     electro_data = []
    #     for electro in range(num_electrodes):
    #         current_signal = data['Signal'][character,:,electro]
    #         timepoints = find_timepoints_1D(current_signal)
    #         sliced = timepoints_slice_second_1D(current_signal,timepoints,seconds_to_slice,Fs)
    #         reshaped_sliced = sliced.reshape([12,15,-1])
    #         electro_data.append(reshaped_sliced)
    #     character_data.append(np.array(electro_data))
    #
    # character_data = character_data.transpose([0,3,2,1,4])
    # return character_data






    # timepoints = find_timepoints_2D(data['StimulusCode'])   # (characters,12*15)
    # character_data = []
    # for character in range(num_characters):
    #     times_data = []
    #     for time in range(15):
    #         intens_data = []
    #         for intens in range(12):
    #             current_timepoint_index = time*12+intens
    #             electro_data = []
    #             for electro in range(64):
    #                 electro_data.append(timepoints_slice_second_1D(data['Signal'][character,:,electro],
    #                                            timepoints[character,current_timepoint_index],
    #                                            seconds_to_slice,Fs))






