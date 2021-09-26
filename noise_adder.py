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
import random

class Room:
    def __init__(self, _id, _roomName):
        self._id = _id
        self._roomName = _roomName

class Rir:
    def __init__(self, _id, _rirName, _roomName, _rirPath):
        self._id = _id
        self._rirName = _rirName
        self._roomName = _roomName
        self._rirPath = _rirPath

class Noise:
    def __init__(self, _id, _noiseName, _bgorfg, _noiseType, _noisePath):
        self._id = _id
        self._noiseName = _noiseName
        self._bgorfg = _bgorfg  # 背景噪音还是前景噪音
        self._noiseType = _noiseType
        self._noisePath = _noisePath


# from typing import Tensor
class NoiseAdder():
    def __init__(self, cfg):
        self.roomList = []
        self.noiseList = []
        self.rirList = []
        self.cfg = cfg
        self.__load_room()
        self.__load_rir()
        self.__load_noise()
        if "RoomInfo" not in cfg:
            logging.error("Error: error room info config")
        if "RirInfo" not in cfg:
            logging.error("Error: error rir info config")
        if "NoiseInfo" not in cfg:
            logging.error("Error: error noise info config")
        if "SnrInfo" not in cfg:
            logging.error("Error: error snr info config")
        if "RirRatio" not in cfg:
            logging.error("Error: error rir ratio config")
        if "NoiseRatio" not in cfg:
            logging.error("Error: error Noise Ratio config")

    def __load_room(self):
        if self.cfg['RoomInfo'] == "":
            logging.error("Error: error room info config")
            return -1
        with open(self.cfg['RoomInfo'], "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                self.roomList.append(Room(len(self.roomList), data[1]))
        print("Info: room info loaded")
        return 0

    def __load_rir(self):
        if self.cfg['RirInfo'] == "":
            print("Error: error rir info config")
            return -2
        with open(self.cfg['RirInfo'], "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if os.access(data[2], os.R_OK) == False:
                    logging.warning(f"Warning: rir file {data[2]} can not be access, skip it")
                self.rirList.append(Rir(len(self.rirList), data[0], data[1], data[2]))
        print("Info: rir info loaded")
        return 0

    def __load_noise(self):
        if self.cfg['NoiseInfo'] == "":
            print("Error: error noise info config")
            return -3
        with open(self.cfg['NoiseInfo'], "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if (data[0][0] == '#'):
                    continue;
                if data[1]!="isotropic" and data[1]!="point-source":
                    logging.warning("Warning: error noise-type(isotropic|point-source) ", data[1])
                    continue;
                if data[2]!="background" and data[2]!="foreground":
                    logging.warning("Warning: error fg-bg-type(background|foreground) ",data[2])
                    continue
                if data[3] != "free":
                    logging.warning("Warning: error noise type) ",data[3])
                    continue
                if os.access(data[4], os.R_OK) == False:
                    logging.warning(f"Warning: noise file {data[4]} can not be access, skip it")
                    continue
                lens = len(data[4])
                if lens <= 0:
                    logging.warning(f"Warning: noise file {data[4]} can not be access, skip it")
                    continue;
                self.noiseList.append(Noise(data[0], data[1], data[2], data[3], data[4]))
        print("Info: noise info loaded")
        return 0
            
    def __chooseRir(self):
        return self.rirList[random.randint(0, len(self.rirList)-1)]

    def __chooseNoise(self):
        return self.noiseList[random.randint(0, len(self.noiseList)-1)]

    def __doNoise(self):
        if(random.random() <= self.cfg['NoiseRatio']):
            return True
        else:
            return False

    def __doReverberation(self):
        if(random.random() <= self.cfg['RirRatio']):
            return True
        else:
            return False   
        
    def __conduct_reverb(self, speech, rir, sample_rate):
        # we convolve the speech signal with the RIR filter.
        rir = rir / torch.norm(rir, p=2)
        rir = torch.flip(rir, [1])
        rir = rir[:, int(0.5*rir.size(1)):]

        speech_ = torch.nn.functional.pad(speech, (rir.shape[1]-1, 0))
        augmented = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]

        return augmented

    def __conduct_noise(self, speech, noise, snr_db, sample_rate):
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> background noise >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        len_noise = noise.size(1)
        len_speech = speech.size(1)
        if(len_noise<len_speech):
            print("noise length error")
            return
        start = random.randint(0, len_noise-len_speech)
        noise = noise[:, start:start+speech.shape[1]]  # Randomly pick a starting position
        
        # print(f"noiseStart: {start}")
        speech_power = speech.norm(p=2)
        noise_power = noise.norm(p=2)

        snr = math.exp(snr_db / 10)
        scale = snr * noise_power / speech_power
        noisy_speech = (scale * speech + noise) / 2

        return noisy_speech

    def add_noise(self, ori_wav):
        # ori_wav -> noise_wav
   
        # add_reverb
        # r_s = time.time()
        if self.cfg['if_add_reverb']==True and self.__doReverberation()==True:
            rir_obj = self.__chooseRir()
            rir_wav, rir_sample_rate = torchaudio.load(rir_obj._rirPath)
            ori_wav = self.__conduct_reverb(ori_wav, rir_wav, rir_sample_rate)
        # r_e = time.time()
        # print(f"adverb_cost_time: {r_e-r_s}")

        # add_noise
        # n_s = time.time()
        if self.cfg['if_add_noise']==True and self.__doNoise()==True:
            noise_obj = self.__chooseNoise()
            noise_wav, noise_sample_rate = torchaudio.load(noise_obj._noisePath)
            snr_db = random.uniform(self.cfg['SnrInfo'][0], self.cfg['SnrInfo'][1])
            ori_wav = self.__conduct_noise(ori_wav, noise_wav, snr_db, noise_sample_rate)
        # n_e = time.time()
        # print(f"noise_cost_time: {n_e-n_s}")
        return ori_wav
