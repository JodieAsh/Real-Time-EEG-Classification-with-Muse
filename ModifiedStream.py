"""
Modified version of the MuseLSL stream function.
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

from pylsl import StreamInfo, StreamOutlet
from functools import partial
from muselsl.muse import Muse
from muselsl.constants import MUSE_SCAN_TIMEOUT, AUTO_DISCONNECT_DELAY,  \
    MUSE_NB_EEG_CHANNELS, MUSE_SAMPLING_EEG_RATE, LSL_EEG_CHUNK,  \
    MUSE_NB_PPG_CHANNELS, MUSE_SAMPLING_PPG_RATE, LSL_PPG_CHUNK, \
    MUSE_NB_ACC_CHANNELS, MUSE_SAMPLING_ACC_RATE, LSL_ACC_CHUNK, \
    MUSE_NB_GYRO_CHANNELS, MUSE_SAMPLING_GYRO_RATE, LSL_GYRO_CHUNK


def stream(address, backend='auto', interface=None, name=None, ppg_enabled=False, acc_enabled=False, gyro_enabled=False, eeg_disabled=False,):
    """
    Begins LSL stream(s) from a Muse with a given address with data sources determined by arguments.
    Uses bluemuse as backend.
    :param address: MAC address of your Muse headband
    """
    bluemuse = backend == 'bluemuse'

    if not eeg_disabled:
        eeg_info = StreamInfo('Muse', 'EEG', MUSE_NB_EEG_CHANNELS, MUSE_SAMPLING_EEG_RATE, 'float32',
                              'Muse%s' % address)
        eeg_info.desc().append_child_value("manufacturer", "Muse")
        eeg_channels = eeg_info.desc().append_child("channels")

        for c in ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']:
            eeg_channels.append_child("channel") \
                .append_child_value("label", c) \
                .append_child_value("unit", "microvolts") \
                .append_child_value("type", "EEG")

        eeg_outlet = StreamOutlet(eeg_info, LSL_EEG_CHUNK)

    if ppg_enabled:
        ppg_info = StreamInfo('Muse', 'PPG', MUSE_NB_PPG_CHANNELS, MUSE_SAMPLING_PPG_RATE,
                              'float32', 'Muse%s' % address)
        ppg_info.desc().append_child_value("manufacturer", "Muse")
        ppg_channels = ppg_info.desc().append_child("channels")

        for c in ['PPG1', 'PPG2', 'PPG3']:
            ppg_channels.append_child("channel") \
                .append_child_value("label", c) \
                .append_child_value("unit", "mmHg") \
                .append_child_value("type", "PPG")

        ppg_outlet = StreamOutlet(ppg_info, LSL_PPG_CHUNK)

    if acc_enabled:
        acc_info = StreamInfo('Muse', 'ACC', MUSE_NB_ACC_CHANNELS, MUSE_SAMPLING_ACC_RATE,
                              'float32', 'Muse%s' % address)
        acc_info.desc().append_child_value("manufacturer", "Muse")
        acc_channels = acc_info.desc().append_child("channels")

        for c in ['X', 'Y', 'Z']:
            acc_channels.append_child("channel") \
                .append_child_value("label", c) \
                .append_child_value("unit", "g") \
                .append_child_value("type", "accelerometer")

        acc_outlet = StreamOutlet(acc_info, LSL_ACC_CHUNK)

    if gyro_enabled:
        gyro_info = StreamInfo('Muse', 'GYRO', MUSE_NB_GYRO_CHANNELS, MUSE_SAMPLING_GYRO_RATE,
                               'float32', 'Muse%s' % address)
        gyro_info.desc().append_child_value("manufacturer", "Muse")
        gyro_channels = gyro_info.desc().append_child("channels")

        for c in ['X', 'Y', 'Z']:
            gyro_channels.append_child("channel") \
                .append_child_value("label", c) \
                .append_child_value("unit", "dps") \
                .append_child_value("type", "gyroscope")

        gyro_outlet = StreamOutlet(gyro_info, LSL_GYRO_CHUNK)

    def push(data, timestamps, outlet):
        for ii in range(data.shape[1]):
            outlet.push_sample(data[:, ii], timestamps[ii])

    push_eeg = partial(push, outlet=eeg_outlet) if not eeg_disabled else None
    push_ppg = partial(push, outlet=ppg_outlet) if ppg_enabled else None
    push_acc = partial(push, outlet=acc_outlet) if acc_enabled else None
    push_gyro = partial(push, outlet=gyro_outlet) if gyro_enabled else None

    if all(f is None for f in [push_eeg, push_ppg, push_acc, push_gyro]):
        print('Stream initiation failed: At least one data source must be enabled.')
        return

    muse = Muse(address=address, callback_eeg=push_eeg, callback_ppg=push_ppg, callback_acc=push_acc, callback_gyro=push_gyro,
                backend=backend, interface=interface, name=name)

    if(bluemuse):
        muse.connect()

    didConnect = muse.connect()

    if(didConnect):
        print('Connected.')
        muse.start()

        eeg_string = " EEG" if not eeg_disabled else ""
        ppg_string = " PPG" if ppg_enabled else ""
        acc_string = " ACC" if acc_enabled else ""
        gyro_string = " GYRO" if gyro_enabled else ""

        print("Streaming%s%s%s%s..." %
              (eeg_string, ppg_string, acc_string, gyro_string))

    return None
