import numpy as np


def averageEnergy(frames):
    energy = np.average(frames * frames, axis=1)
    return energy.reshape(-1)


def zeroCrossingRate(frames):
    _, lens = frames.shape
    delta = np.abs(np.sign(frames[:, 1:]) - np.sign(frames[:, :lens - 1]))
    zeroCrossingRate = np.average(delta / 2, axis=1)
    return zeroCrossingRate.reshape(-1)


def std(frames):
    return np.std(frames, axis=1).reshape(-1)


def kurt(frames):
    maximum = np.max(frames, axis=1)
    rms = np.sqrt(averageEnergy(frames)) + 1e-6
    return (maximum / rms).reshape(-1)


def wave(frames):
    rms = np.sqrt(averageEnergy(frames)) + 1e-6
    mean = np.average(frames, axis=1)
    mean[abs(mean) < 1e-6] = 1e-6
    return (rms / mean).reshape(-1)


def grad(frames):
    _, lens = frames.shape
    delta = np.abs(frames[:, 1:] - frames[:, :lens - 1])
    return np.average(delta, axis=1).reshape(-1)


def relate(frames):
    lens, _ = frames.shape
    product = frames[:lens - 1, :] * frames[1:, :]
    return np.average(product, axis=1).reshape(-1)
