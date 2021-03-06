import os
import numpy as np
import config
import utils
import feature
from tqdm import tqdm
import pickle
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import logfbank

datadict = {}
for group in tqdm(os.listdir(config.datasetPath)):
    for num in range(10):
        numdir = os.path.join(config.datasetPath, group, str(num))
        for file in os.listdir(numdir):
            # 读wav文件并归一化
            info, wave_data = utils.wavDecode(os.path.join(numdir, file))
            (rate,sig) = wav.read(os.path.join(numdir, file))
            if info[2] != 44100:
                wave_data = utils.resample(wave_data, info[2], 44100)
            wave_data = wave_data[:, 0]
            wave_data = wave_data * 1.0 / (max(abs(wave_data)))
            wave_data = utils.double_thresh(wave_data)
            mfcc_feat = mfcc(sig,rate)
            print(mfcc_feat)
#             if len(wave_data) < 1000:
#                 print(os.path.join(numdir, file))

#             # 帧数
#             frames = utils.split(wave_data, 10)
#             nf = frames.shape[0]
#             frames = utils.window(frames)

#             # 特征
#             feat = [
#                 feature.averageEnergy(frames).reshape((1, -1)),
#                 feature.std(frames).reshape((1, -1)),
#                 feature.kurt(frames).reshape((1, -1)),
#                 feature.wave(frames).reshape((1, -1)),
#                 feature.zeroCrossingRate(frames).reshape((1, -1)),
#                 feature.grad(frames).reshape((1, -1)),
#                 feature.relate(frames).reshape((1, -1))
#             ]
#             feat = np.hstack(feat)

#             if num not in datadict.keys():
#                 datadict[num] = []
#             datadict[num].append(feat)

# x = []
# y = []

# for num in range(10):
#     for data in datadict[num]:
#         x.append(data)
#         y.append(num)

# with open("data/x", "wb") as f:
#     pickle.dump(x, f)
# with open("data/y", "wb") as f:
#     pickle.dump(y, f)
