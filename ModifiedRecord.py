"""
Modified version of the MuseLSL record function.
Copyright © 2018, authors of muselsl - All rights reserved.
Version History:
2019, May:
   Original script by Alexandre Barachant, Dano Morrison, Hubert Banville, Jason Kowaleski ,Uri Shaked,
   Sylvain Chevallier and Juan Jesús Torre Tresols.
  doi = 10.5281/zenodo.3228861,
  url = https://doi.org/10.5281/zenodo.3228861
2019:
    Modified by Jodie Ashford, Aston University
"""

import numpy as np
import pandas as pd
import os
from pylsl import StreamInlet, resolve_byprop
from sklearn.linear_model import LinearRegression
from time import time, strftime, gmtime
from muselsl. constants import LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK, LSL_PPG_CHUNK, LSL_ACC_CHUNK, LSL_GYRO_CHUNK


def start_stream(data_source="EEG", chunk_length=LSL_EEG_CHUNK):
    """
    Connects to an existing Muse LSL stream
    :return: inlet: a new stream inlet from a resolved stream description
    """
    print("Looking for a %s stream..." % data_source)
    streams = resolve_byprop('type', data_source, timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        print("Can't find %s stream." % data_source)
        return

    print("Started acquiring data.")
    inlet = StreamInlet(streams[0], max_chunklen=chunk_length)
    return inlet


def record_csv(duration, inlet, filename=None, dejitter=False, data_source="EEG"):
    """
    Records a fixed duration of EEG data from an LSL stream into a CSV file
    Note: there must be an existing Muse LSL stream to record from

    :param duration: The amount of time (seconds) to record data for.
    :param inlet: A resolved stream description object.
    :param filename: The name of the resulting CSV file.
    :return: A CSV file containing the recorded EEG data.
    """

    chunk_length = LSL_EEG_CHUNK
    if data_source == "PPG":
        chunk_length = LSL_PPG_CHUNK
    if data_source == "ACC":
        chunk_length = LSL_ACC_CHUNK
    if data_source == "GYRO":
        chunk_length = LSL_GYRO_CHUNK

    if not filename:
        filename = os.path.join(
            os.getcwd(),
            # "output_files",
            "%s_recording_%s.csv" % (data_source, strftime('%Y-%m-%d-%H.%M.%S', gmtime())))

    # eeg_time_correction = inlet.time_correction()

    info = inlet.info()
    description = info.desc()

    Nchan = info.channel_count()

    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, Nchan):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    res = []
    timestamps = []
    t_init = time()
    # time_correction = inlet.time_correction()
    print('Start recording at time t=%.3f' % t_init)
    # print('Time correction: ', time_correction)
    while (time() - t_init) < duration:
        try:
            data, timestamp = inlet.pull_chunk(timeout=1.0,
                                               max_samples=chunk_length)

            if timestamp:
                res.append(data)
                timestamps.extend(timestamp)

        except KeyboardInterrupt:
            break

    time_correction = inlet.time_correction()
    # print('Time correction: ', time_correction)

    res = np.concatenate(res, axis=0)
    timestamps = np.array(timestamps) + time_correction

    if dejitter:
        y = timestamps
        X = np.atleast_2d(np.arange(0, len(y))).T
        lr = LinearRegression()
        lr.fit(X, y)
        timestamps = lr.predict(X)

    res = np.c_[timestamps, res]
    data = pd.DataFrame(data=res, columns=['timestamps'] + ch_names)

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    data.to_csv(filename, float_format='%.3f', index=False)

    print('Done - wrote file: ' + filename + '.')

    # return filename


def record_numpy(duration, inlet, dejitter=False, data_source="EEG"):
    """
    Records a fixed duration of EEG data from an LSL stream into a numpy array.
    Note: there must be an existing muselsl stream.
    :param duration: The amount of time (seconds) to record data for.
    :param inlet: A resolved stream description object.
    :return: A numpy array containing the recorded EEG data.
    """
    chunk_length = LSL_EEG_CHUNK
    if data_source == "PPG":
        chunk_length = LSL_PPG_CHUNK
    if data_source == "ACC":
        chunk_length = LSL_ACC_CHUNK
    if data_source == "GYRO":
        chunk_length = LSL_GYRO_CHUNK

    info = inlet.info()
    description = info.desc()

    Nchan = info.channel_count()

    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, Nchan):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    res = []
    timestamps = []
    t_init = time()
    #print('Start recording at time t=%.3f' % t_init)
    while (time() - t_init) < duration:
        try:
            data, timestamp = inlet.pull_chunk(timeout=1.0,
                                               max_samples=chunk_length)

            if timestamp:
                res.append(data)
                timestamps.extend(timestamp)

        except KeyboardInterrupt:
            break

    time_correction = inlet.time_correction()

    res = np.concatenate(res, axis=0)
    timestamps = np.array(timestamps) + time_correction

    if dejitter:
        y = timestamps
        X = np.atleast_2d(np.arange(0, len(y))).T
        lr = LinearRegression()
        lr.fit(X, y)
        timestamps = lr.predict(X)

    res = np.c_[timestamps, res]
    data = pd.DataFrame(data=res, columns=['timestamps'] + ch_names)

    numpy_array = data.to_numpy()
    # print('Finished - converted data to numpy array ')
    return numpy_array
