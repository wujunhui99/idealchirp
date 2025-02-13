from pyLoRa import pyLoRa
import numpy as np
from scipy import signal









def fft_lowpass_filter(sig, fs, cutoff):
    # sig: 输入信号(复数numpy数组)
    # fs: 采样率 1MHz
    # cutoff: 截止频率 62.5kHz

    # 进行FFT
    fft_sig = np.fft.fft(sig)
    freq = np.fft.fftfreq(len(sig), 1 / fs)

    # 构建频域滤波器
    mask = np.abs(freq) <= cutoff

    # 应用滤波器
    fft_sig_filtered = fft_sig * mask

    # 进行IFFT
    filtered_sig = np.fft.ifft(fft_sig_filtered)

    return filtered_sig


def bandpass_filter(self,sig, fs, cutoff, offset):

    # 进行FFT
    fft_sig = np.fft.fft(sig)
    freq = np.fft.fftfreq(len(sig), 1 / fs)

    # 构建带通滤波器掩码
    mask = np.abs(freq - offset) <= cutoff

    # 应用滤波器
    fft_sig_filtered = fft_sig * mask

    # 进行IFFT
    # filtered_sig = np.fft.ifft(fft_sig_filtered)

    return fft_sig_filtered

# def our_ideal_decode_decodev3(self, sig):
#     filtered_sig = self.gen_filter_matrix(sig)
#     downchirps = self.gen_ideal_matrix()
#     result = np.sum(filtered_sig * downchirps, axis=1)
#     vals = np.abs(result) ** 2
#     est = np.argmax(vals).item()
#     max_val = np.max(vals).item()
#     return est, max_val
# def gen_filter_matrix(self, sig):
#     matrix = np.zeros((2**self.sf, len(sig)),dtype=np.complex64)
#     for i in range(2**self.sf):
#         offset = self.bw * i / 2**self.sf
#         #  bandpass_filter(self,sig, fs, cutoff, offset):
#         bandfilter = self.bandpass_filter(sig, self.fs,cutoff=self.bw, offset=offset)
#         matrix[i] = bandfilter
#     return matrix


    pass
pyLoRa.bandpass_filter = bandpass_filter
pyLoRa.our_ideal_decode_decodev3 = our_ideal_decode_decodev3

lora = pyLoRa()
sig = lora.read_file("./ideal/120.cfile")
print(lora.our_ideal_decode_decodev2(sig))
print(lora.our_ideal_decode_decodev3(sig,lora.ideal_chirp(f0=0,iq_invert=1)))
