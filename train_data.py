import random
import glob
import pprint
import os
import linecache

import utils
import numpy as np
from downloader import reciters

def get_file_list(num_train_samples, num_test_samples, shuffle=True):
    assert num_train_samples + num_test_samples <= len(reciters)
    train_files = []
    test_files  = []
    surah = glob.glob('wav/*/*')
    for s in sorted(surah):
        rec = sorted(glob.glob(s + '/*.wav'))
        if shuffle: random.shuffle(rec)
        train_files += rec[:num_train_samples]
        test_files  += rec[num_train_samples:num_train_samples+num_test_samples]
    return train_files, test_files

def get_transcript(wav_filename):
    trans_file = os.path.dirname(wav_filename) + "/transcript.txt"
    trans_idx  = int(wav_filename.split('_')[-1].replace('.wav', ''))
    return linecache.getline(trans_file, trans_idx+1).strip()

def prepare_inputs(wav_filenames):
    inputs = []
    for wav in wav_filenames:
        mfcc = utils.wav_mfcc(wav)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize
        inputs.append(mfcc)
    train_inputs = np.asarray(inputs)
    return train_inputs

def prepare_targets(wav_filenames):
    targets = []
    num_samples = len(wav_filenames)
    for wav in wav_filenames:
        transcript = get_transcript(wav)
        encoded, _ = utils.encode_target(transcript)
        targets.append(encoded)
    train_targets = np.asarray(targets)
    return train_targets

if __name__ == '__main__':

    train_files, test_files = get_file_list(5, 2)
    pprint.pprint(train_files)

    targets = prepare_targets(train_files)
    pprint.pprint(targets)
