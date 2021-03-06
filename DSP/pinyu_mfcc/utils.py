import numpy as np
import wave
import feature
from numba import jit


def wavDecode(dir: str) -> np.ndarray:
    """
    输入文件位置，返回解析后的序列
    注意返回为二维数组，第一维指定声道
    """
    try:
        f: wave.Wave_read = wave.open(dir, "rb")
    except BaseException:
        print("读取wav文件" + str(dir) + "错误")
        print("请检查路径是否正确")
        return

    cSz, binSz, rate, n, _, _ = f.getparams()
    # 声道数、位深、采样率、采样点数
    info = (cSz, binSz, rate, n)
    tmp = f.readframes(n)

    data: np.ndarray = np.frombuffer(tmp, dtype=np.short)
    data = data.reshape((n, cSz))
    return (info, data)


def frame(wave_data, wlen, inc):
    # 帧数
    signal_length = len(wave_data)
    if signal_length <= wlen:
        nf = 1
    else:
        nf = int(np.ceil((1.0 * signal_length - wlen + inc) / inc))

    # 补齐后分帧
    pad_length = int((nf - 1) * inc + wlen)
    zeros = np.zeros((pad_length - signal_length,))
    pad_signal = np.concatenate((wave_data, zeros))
    indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]
    return frames


def split(wave_data, tickNum):
    signal_length = len(wave_data)
    step = int(np.ceil(signal_length / tickNum))

    pad_length = int(step * tickNum)
    zeros = np.zeros((pad_length - signal_length,))
    pad_signal = np.concatenate((wave_data, zeros))
    frames = pad_signal.reshape((tickNum, -1))
    return frames


def window(frames, method="hanning"):
    lframe = frames.shape[1]
    if method == "hanning":
        windown = np.hanning(lframe)
        return frames * windown


@jit(nopython=True)
def resample(s: np.ndarray, f: int, t: int) -> np.ndarray:
    """
    重采样
    给定原序列、原采样率、目标采样率
    返回重采样后序列
    采用线性插值
    """
    n, m = s.shape

    nn = n * t // f
    ret = np.zeros((nn, m), s.dtype)

    j = 0
    for i in range(nn):
        # 逐点采样
        aT = i / t
        while (j + 1) / f <= aT:
            j += 1
        tmp = i * f / t - j

        if j + 1 > n:
            print("存在边界问题")
            print(nn, i, f, t)
            break

        for k in range(m):
            ret[i, k] = s[j, k] * (1 - tmp) + (s[j + 1,
                                                 k] if j + 1 < n else 0) * tmp

    return ret


def double_thresh(wave_data):
    frame_lens = 512
    move = 128

    frames = frame(wave_data, frame_lens, move)
    energy = feature.averageEnergy(frames)

    energy_mean = energy.mean()
    T1 = np.mean(energy[:10])
    T2 = energy_mean / 4  # 较高的能量阈值
    T1 = (T1 + T2) / 4  # 较低的能量阈值

    range_o = np.arange(len(energy))
    # 首先利用较大能量阈值 MH 进行初步检测
    mask1 = energy > T2
    range1 = range_o[mask1]
    N3, N4 = range1[0], range1[-1]

    # 利用较小能量阈值 ML 进行第二步能量检测
    N2, N5 = N3, N4
    for i in range_o[:N3][::-1]:  # 从N3向左搜索 从N4向右搜索
        if energy[i] <= T1:
            N2 = i
            break
    for j in range_o[N4:]:
        if energy[j] <= T1:
            N5 = j
            break

    L_w = N2 * move + frame_lens // 2
    R_w = N5 * move + frame_lens // 2

    return wave_data[L_w:R_w]
