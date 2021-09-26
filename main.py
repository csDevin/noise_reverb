import yaml
import math
import time
import os
import logging
import torch
import argparse
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from tools import plot_waveform, plot_specgram, play_audio, inspect_file
from noise_adder import NoiseAdder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test asr')
    parser.add_argument('--config', default='noise.yaml', type=str, help='path to config file')
    parser.add_argument('--if-add-reverb', default='True', type=bool, help='whether to add reverberation')
    parser.add_argument('--if-add-noise', default='True', type=bool, help='whether to add noise')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in config.items():
            setattr(args, key, value)
    args.SnrInfo = [item for item in map(float, args.SnrInfo.strip().split())]
    cfg = vars(args)

    wav_dir = "/home/dingchaoyue/code/wenet/examples/aishell/s0/data/33/data_aishell/wav/train/S0002/"
    path_list = os.listdir(wav_dir)
    path_list = path_list[:5]
    
    '''define a object'''
    n_obj = NoiseAdder(cfg)
    
    for curr_wav in path_list:
        input_speech_path = wav_dir+curr_wav
        input_speech, sample_rate = torchaudio.load(input_speech_path)
        # inspect_file(input_speech_path)

        a = time.time()
        
        '''call an object'''
        output_speech = n_obj.add_noise(input_speech)
        
        b = time.time()
        print(f"full_cost_time: {b-a}")
        path = "./output.wav"
        torchaudio.save(path, output_speech, sample_rate)
        # inspect_file(path)
