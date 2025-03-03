import math
from scipy.signal import chirp
from scipy.fft import fft as np_fft
from scipy.signal import chirp as np_chirp
import matplotlib.pyplot as plt
import numpy as np
class PyLoRa:
    def __init__(self, sf=7, bw=125e3,iq_invert=0 ,fs=1e6,sig = None, zero_padding = 10,payload=None, f0=0, preamble_len=6,raw_chirp=None,rf_freq = 915e6, bit = 5):
        if not isinstance(sf, int) or sf < 7 or sf > 12:
            raise ValueError("SF must be an integer between 7 and 12")
        if bw <= 0:
            raise ValueError("Bandwidth must be positive")
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if not isinstance(preamble_len, int) or preamble_len < 0:
            raise ValueError("Preamble length must be a non-negative integer")
        self.sf = sf
        self.bw = bw
        self.fs = fs
        self.f0 = f0
        self.bit = bit
        self.raw_chirp = raw_chirp
        self.sig = sig
        self.payload = payload if payload is not None else []
        self.preamble_len = preamble_len
        self.iq_invert = iq_invert
        self.zero_padding_ratio = zero_padding
        self.symbol_cnt = 0
        self.cfo = 0
        self.sfo = 0
        self.rf_freq = rf_freq
        self.os_ratio =int( self.fs / self.bw)
        self.bin_num = 2 ** self.sf * zero_padding
        self.sfo_accum = 0

    def get_symbol_period(self):
        return (2 ** self.sf) / self.bw

    def get_samples_per_symbol(self):
        return int((2 ** self.sf) * self.fs / self.bw)

    def ideal_chirp(self, f0=0, iq_invert=0):
        self.iq_invert = iq_invert
        num_symbols = 2 ** self.sf
        num_samples = int(num_symbols * self.fs / self.bw)

        # Time array
        t = np.linspace(0, num_symbols / self.bw, num_samples + 1)[:-1]

        # Calculate frequency shift
        freq_shift = (f0 * self.bw) / num_symbols

        # Generate upchirp
        f0_shifted = -self.bw / 2 + freq_shift
        f1_shifted = self.bw / 2 + freq_shift

        # Generate I and Q components
        chirp_i = chirp(t, f0=f0_shifted, f1=f1_shifted, t1=num_symbols / self.bw, method='linear', phi=90)
        chirp_q = chirp(t, f0=f0_shifted, f1=f1_shifted, t1=num_symbols / self.bw, method='linear', phi=0)
        if self.iq_invert:
            chirp_q = - chirp_q
        # Create complex signal
        signal = chirp_i + 1j * chirp_q

        # Normalize
        self.raw_chirp = signal
        return signal








    def add_noise(self, sig, snr):
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol
        # add noise of a certain SNR, chosen from snr_range
        amp = math.pow(0.1, snr / 20) * np.mean(np.abs(sig))
        noise = (amp / math.sqrt(2) * np.random.randn(num_samples) + 1j * amp / math.sqrt(2) * np.random.randn(
            num_samples))
        dataX = sig + noise  # dataX: data with noise
        return dataX


    def gen_ideal_matrix(self):
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol
        result = np.zeros((num_classes, num_samples), dtype=np.complex64)
        for i in range(num_classes):
            result[i] = self.ideal_chirp(f0=i,iq_invert=1)
            self.f0 = i
        return result

    def our_ideal_decode_decodev2z(self,sig):
        downchirp = self.ideal_chirp(f0=0,iq_invert=1)
        dechirped = sig * downchirp
        fft_raw = np_fft(dechirped, len(dechirped))
        useful = fft_raw[:self.get_samples_per_symbol()]
        mag = np.abs(useful) ** 2
        est = np.argmax(mag)
        max_val = np.max(mag)
        return est, max_val
    def our_ideal_decode_decodev2(self, sig):
        mtx = self.gen_ideal_matrix()
        sig = np.array(sig).T
        result = np.matmul(mtx, sig)
        vals = np.abs(result) ** 2
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est, max_val

    def gen_ideal_matrix_bit(self,bit):
        num_classes = 2 ** bit  # number of codes per symbol == 2 ** sf
        num_samples = int(2 ** self.sf * self.fs / self.bw)  # number of samples per symbol
        result = np.zeros((num_classes, num_samples), dtype=np.complex64)
        for i in range(num_classes):
            result[i] = self.ideal_chirp(f0=i * 2 ** (self.sf - bit),iq_invert=1)
            self.f0 = i * 2 ** (self.sf - bit)
        return result

if __name__ == '__main__':
    lora = PyLoRa()
    sig = lora.ideal_chirp(f0=16)
    sig = lora.add_noise(sig=sig,snr=-25)
    lora.our_ideal_decode_decodev2(sig=sig)






























