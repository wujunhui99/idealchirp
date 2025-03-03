import math
from scipy.signal import chirp
from scipy.fft import fft as np_fft
from scipy.signal import chirp as np_chirp
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from cupyx.scipy.fft import fft as cp_fft
from cupyx.scipy import signal as cp_signal
import time


class PyLoRa:
    def __init__(self, sf=7, bw=125e3, iq_invert=0, fs=1e6, sig=None, zero_padding=10, payload=None, f0=0,
                 preamble_len=6, raw_chirp=None, rf_freq=915e6, bit=5, use_gpu=True):
        # Add GPU flag
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
        self.os_ratio = int(self.fs / self.bw)
        self.bin_num = 2 ** self.sf * zero_padding
        self.sfo_accum = 0
        self.use_gpu = use_gpu

        # Try to initialize CuPy and check GPU availability
        if self.use_gpu:
            try:
                cp.cuda.runtime.getDeviceCount()
                print("GPU acceleration enabled")
            except:
                print("No CUDA-compatible GPU found, falling back to CPU")
                self.use_gpu = False

    # Modified helper method to handle array transfer between CPU and GPU
    def to_gpu(self, arr):
        if self.use_gpu and not isinstance(arr, cp.ndarray):
            return cp.asarray(arr)
        return arr

    def to_cpu(self, arr):
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr

    def get_symbol_period(self):
        return (2 ** self.sf) / self.bw

    def get_samples_per_symbol(self):
        return int((2 ** self.sf) * self.fs / self.bw)

    def ideal_chirp(self, f0=0, iq_invert=0):
        self.iq_invert = iq_invert
        num_symbols = 2 ** self.sf
        num_samples = int(num_symbols * self.fs / self.bw)

        # Time array
        if self.use_gpu:
            t = cp.linspace(0, num_symbols / self.bw, num_samples + 1)[:-1]
        else:
            t = np.linspace(0, num_symbols / self.bw, num_samples + 1)[:-1]

        # Calculate frequency shift
        freq_shift = (f0 * self.bw) / num_symbols

        # Generate upchirp
        f0_shifted = -self.bw / 2 + freq_shift
        f1_shifted = self.bw / 2 + freq_shift

        # Generate I and Q components (using appropriate library)
        if self.use_gpu:
            # Create a GPU implementation of chirp
            chirp_i = cp.cos(
                cp.pi * (f0_shifted + (f1_shifted - f0_shifted) * t / (num_symbols / self.bw)) * t + cp.pi / 2)
            chirp_q = cp.cos(cp.pi * (f0_shifted + (f1_shifted - f0_shifted) * t / (num_symbols / self.bw)) * t)
        else:
            chirp_i = chirp(t, f0=f0_shifted, f1=f1_shifted, t1=num_symbols / self.bw, method='linear', phi=90)
            chirp_q = chirp(t, f0=f0_shifted, f1=f1_shifted, t1=num_symbols / self.bw, method='linear', phi=0)

        if self.iq_invert:
            chirp_q = -chirp_q

        # Create complex signal
        signal = chirp_i + 1j * chirp_q

        # Normalize
        self.raw_chirp = signal
        return signal
    def our_ideal_decode(self, sig):
        down_chirp = self.ideal_chirp(f0=0, iq_invert=1)
        dechirp = sig * down_chirp
        # Compute FFT
        fft_result = np.fft.fft(dechirp)

        # Get magnitude spectrum (absolute values)
        magnitudes = np.abs(fft_result)



        max_magnitude = np.max(magnitudes)
        max_idx = np.argmax(magnitudes)
        return max_idx, max_magnitude


    def reverse_arr(self,iq_data,beg,end):
        while(end > beg):
            iq_data[beg],iq_data[end] = iq_data[end],iq_data[beg]
            end -= 1
            beg += 1

    def real_chirp(self, f0=0, iq_invert=0):
        ideal = self.ideal_chirp(f0 = 0,iq_invert = iq_invert)
        index = int((f0 * self.get_samples_per_symbol()) / (2**self.sf))
        num_sample = self.get_samples_per_symbol()
        random_phase1 = np.random.uniform(0, 2 * np.pi)  # 单个随机值
        phase_shift1 = np.exp(1j * random_phase1)
        random_phase2 = np.random.uniform(0, 2 * np.pi)  # 单个随机值
        phase_shift2 = np.exp(1j * random_phase2)
        ideal[:index] = ideal[:index] * phase_shift1
        self.reverse_arr(ideal,0,index-1)
        ideal[index:] = ideal[index:] * phase_shift2
        self.reverse_arr(ideal,index,num_sample-1)
        self.reverse_arr(ideal,0 ,num_sample-1)
        return ideal


    def save_chirp(self, iq_data,remarks = "", f0=0, iq_invert=0):


        iq_data = iq_data.astype(np.complex64)
        fsM = str(int(self.fs/1e6))  + "M" if  self.fs / 1e6 else  ""
        fsK =  str((int(self.fs % 1e6)/1e3 )) + "K" if  self.fs % 1e6 / 1e3 else ""
        fs = fsM + fsK
        filename = remarks + "chirp"+ "f" + str(self.f0) + "sf" + str(self.sf) + "bw" + str(int(self.bw/1000)) + "k"  + "fs" + fs+ "invert" + str(self.iq_invert) + ".cfile"
        iq_data.tofile(filename)

    def add_noise(self, sig, snr):
        sig = self.to_gpu(sig)
        num_classes = 2 ** self.sf
        num_samples = int(num_classes * self.fs / self.bw)

        if self.use_gpu:
            amp = cp.power(0.1, snr / 20) * cp.mean(cp.abs(sig))
            noise = (amp / cp.sqrt(2) * cp.random.randn(num_samples) +
                     1j * amp / cp.sqrt(2) * cp.random.randn(num_samples))
            dataX = sig + noise
            return self.to_cpu(dataX)
        else:
            amp = math.pow(0.1, snr / 20) * np.mean(np.abs(sig))
            noise = (amp / math.sqrt(2) * np.random.randn(num_samples) +
                     1j * amp / math.sqrt(2) * np.random.randn(num_samples))
            dataX = sig + noise
            return dataX
    def add_noise2(self, sig, snr):
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol
        # add noise of a certain SNR, chosen from snr_range
        amp = math.pow(0.1, snr / 10) * np.mean(np.abs(sig))
        noise = (amp  * np.random.randn(num_samples) + 1j * amp  * np.random.randn(
            num_samples))
        dataX = sig + noise  # dataX: data with noise
        return dataX
    def plot_IQsignal(self, chirp_signal=None, f0=0, iq_invert=0):
        if chirp_signal is None:
            chirp_signal = self.real_chirp(f0, iq_invert)
        plt.figure(figsize=(12, 4))
        t = np.arange(len(chirp_signal)) / self.fs * 1000  # ms
        plt.plot(t,  chirp_signal.real, label='Real')
        plt.plot(t,  chirp_signal.imag, label='Imag')
        freq_offset = f0 * self.bw / (2 ** self.sf) / 1e3  # kHz
        plt.title(f'Chirp with f0={f0} (freq offset={freq_offset:.2f} kHz)')
        plt.xlabel('Time (ms)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def loraphy(self, sig):
        sig = self.to_gpu(sig)
        upsampling = 100  # up-sampling rate for loraphy, default 100
        num_classes = 2 ** self.sf

        # dechirp
        downchirp = self.ideal_chirp(f0=0, iq_invert=1)
        chirp_data = sig * downchirp

        # compute FFT
        if self.use_gpu:
            fft_raw = cp_fft(chirp_data, int(len(chirp_data) * upsampling))

            # cut the FFT results to two (due to upsampling)
            target_nfft = num_classes * upsampling
            cut1 = fft_raw[:target_nfft]
            cut2 = fft_raw[-target_nfft:]
            combined_spectrum = cp.abs(cut1) + cp.abs(cut2)

            # add absolute values of cut1 and cut2 to merge two peaks into one
            peak_idx = int(round((cp.argmax(combined_spectrum) / upsampling).item()) % num_classes)
            peak_value = float(cp.max(combined_spectrum).item())
        else:
            fft_raw = np_fft(chirp_data, len(chirp_data) * upsampling)

            # cut the FFT results to two (due to upsampling)
            target_nfft = num_classes * upsampling
            cut1 = fft_raw[:target_nfft]
            cut2 = fft_raw[-target_nfft:]
            combined_spectrum = np.abs(cut1) + np.abs(cut2)

            # add absolute values of cut1 and cut2 to merge two peaks into one
            peak_idx = round((np.argmax(combined_spectrum) / upsampling).item()) % num_classes
            peak_value = np.max(combined_spectrum)

        return peak_idx, peak_value


    def our_ideal_decode_decodev2z(self,sig):
        downchirp = self.ideal_chirp(f0=0,iq_invert=1)
        dechirped = sig * downchirp
        fft_raw = np_fft(dechirped, len(dechirped))
        useful = fft_raw[:self.get_samples_per_symbol()]
        mag = np.abs(useful) ** 2
        est = np.argmax(mag)
        max_val = np.max(mag)
        return est, max_val

    def our_ideal_decode_decodev2x(self, sig):
        mtx = self.gen_tone_matrix()
        sig = sig * self.ideal_chirp(f0=0,iq_invert=1)
        sig = np.array(sig).T
        result = np.matmul(mtx, sig)
        vals = np.abs(result) ** 2
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est, max_val

    def gen_tone_matrix(self):
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol
        result = np.zeros((num_classes, num_samples), dtype=np.complex64)
        for i in range(num_classes):
            result[i] = self.gen_tone(freq=- i * self.bw / num_classes)
        return result

    def bandpass_filter(self, fft_sig, fs, cutoff, offset):

        # 进行FFT
        freq = np.fft.fftfreq(len(fft_sig), 1 / fs)
        # 构建带通滤波器掩码
        mask = np.abs(freq - offset) <= cutoff
        # 应用滤波器
        fft_sig_filtered = fft_sig * mask
        return fft_sig_filtered
    def our_ideal_decode_decodev3(self, up):
        # 对两个信号做傅里叶变换
        down = self.ideal_chirp(f0=0,iq_invert=1)
        up_freq = np.fft.fft(up)
        down_freq = np.fft.fft(down)

        # 在频域上进行卷积操作
        N = len(up_freq)
        res = np.zeros(N, dtype=complex)

        for i in range(N):
            offset = self.bw * i / 2 ** self.sf
            bandfilter = self.bandpass_filter(up_freq, self.fs, cutoff=self.bw, offset=offset)
            for j in range(N):
                k = (i + j) % N  # 循环卷积
                res[k] += bandfilter[i] * down_freq[j] / N
        res = np.abs(res[:2 ** self.sf])

        value = np.max(res)
        index = np.argmax(res)
        return index, value
    def gen_tone(self, freq):
        length = self.get_samples_per_symbol()
        fs = self.fs
        t = np.arange(length) / fs
        signal = np.exp(1j * 2 * np.pi * freq * t)
        return signal





    def fft_lowpass_filter(self,sig, fs, cutoff):
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
    def filter2_loratrimmer_decode(self, sig):
        dataE1,dataE2 = self.gen_constants()
        sig = self.fft_lowpass_filter(sig, self.fs, self.bw/2)
        sig = np.array(sig).T
        data1 = np.matmul(dataE1, sig)
        data2 = np.matmul(dataE2, sig)
        vals = np.abs(data1) ** 2 + np.abs(data2) ** 2
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est, max_val
    def filter_loratrimmer_decode(self, sig):
        dataE1,dataE2 = self.gen_constants()
        sig = self.fft_lowpass_filter(sig, self.fs, self.bw)
        sig = np.array(sig).T
        data1 = np.matmul(dataE1, sig)
        data2 = np.matmul(dataE2, sig)
        vals = np.abs(data1) ** 2 + np.abs(data2) ** 2
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est, max_val

    def loratrimmer_decode(self, sig):
        sig = self.to_gpu(sig)
        dataE1, dataE2 = self.gen_constants()

        if self.use_gpu:
            sig = cp.array(sig).T
            data1 = cp.matmul(dataE1, sig)
            data2 = cp.matmul(dataE2, sig)
            vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
            est = int(cp.argmax(vals).item())
            max_val = float(cp.max(vals).item())
        else:
            sig = np.array(sig).T
            data1 = np.matmul(dataE1, sig)
            data2 = np.matmul(dataE2, sig)
            vals = np.abs(data1) ** 2 + np.abs(data2) ** 2
            est = np.argmax(vals).item()
            max_val = np.max(vals).item()

        return est, max_val

    def gen_constants(self):
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol

        # generate downchirp
        if self.use_gpu:
            t = cp.linspace(0, num_samples / self.fs, num_samples + 1)[:-1]

            # Create a GPU implementation of chirp for downchirp
            phase1 = cp.pi * (self.bw / 2 + (-self.bw - self.bw / 2) * t / (2 ** self.sf / self.bw)) * t + cp.pi / 2
            phase2 = cp.pi * (self.bw / 2 + (-self.bw - self.bw / 2) * t / (2 ** self.sf / self.bw)) * t

            chirpI1 = cp.cos(phase1)
            chirpQ1 = cp.cos(phase2)
            downchirp = chirpI1 + 1j * chirpQ1

            # two DFT matrices
            dataE1 = cp.zeros((num_classes, num_samples), dtype=cp.complex64)
            dataE2 = cp.zeros((num_classes, num_samples), dtype=cp.complex64)
        else:
            t = np.linspace(0, num_samples / self.fs, num_samples + 1)[:-1]

            chirpI1 = np.array(
                np_chirp(t, f0=self.bw / 2, f1=-self.bw / 2, t1=2 ** self.sf / self.bw, method='linear', phi=90))
            chirpQ1 = np.array(
                np_chirp(t, f0=self.bw / 2, f1=-self.bw / 2, t1=2 ** self.sf / self.bw, method='linear', phi=0))
            downchirp = chirpI1 + 1j * chirpQ1

            # two DFT matrices
            dataE1 = np.zeros((num_classes, num_samples), dtype=np.complex64)
            dataE2 = np.zeros((num_classes, num_samples), dtype=np.complex64)

        for symbol_index in range(num_classes):
            time_shift = int(symbol_index / num_classes * num_samples)
            time_split = num_samples - time_shift
            dataE1[symbol_index][:time_split] = downchirp[time_shift:]
            if symbol_index != 0:
                dataE2[symbol_index][time_split:] = downchirp[:time_shift]

        return dataE1, dataE2

    def our_ideal_decode_decodev2(self, sig):
        sig = self.to_gpu(sig)
        mtx = self.gen_ideal_matrix()

        if self.use_gpu:
            sig = cp.array(sig).T
            result = cp.matmul(mtx, sig)
            vals = cp.abs(result) ** 2
            est = int(cp.argmax(vals).item())
            max_val = float(cp.max(vals).item())
        else:
            sig = np.array(sig).T
            result = np.matmul(mtx, sig)
            vals = np.abs(result) ** 2
            est = np.argmax(vals).item()
            max_val = np.max(vals).item()

        return est, max_val

    def gen_ideal_matrix(self):
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol

        if self.use_gpu:
            result = cp.zeros((num_classes, num_samples), dtype=cp.complex64)
        else:
            result = np.zeros((num_classes, num_samples), dtype=np.complex64)

        for i in range(num_classes):
            result[i] = self.ideal_chirp(f0=i, iq_invert=1)
            self.f0 = i
        return result
    def our_ideal_decode_decodev2_bit(self, sig):
        mtx = self.gen_ideal_matrix_bit(self.bit)
        sig = np.array(sig).T
        result = np.matmul(mtx, sig)
        vals = np.abs(result) ** 2
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est , max_val
    def gen_ideal_matrix_bit(self,bit):
        num_classes = 2 ** bit  # number of codes per symbol == 2 ** sf
        num_samples = int(2 ** self.sf * self.fs / self.bw)  # number of samples per symbol
        result = np.zeros((num_classes, num_samples), dtype=np.complex64)
        for i in range(num_classes):
            result[i] = self.ideal_chirp(f0=i * 2 ** (self.sf - bit),iq_invert=1)
            self.f0 = i * 2 ** (self.sf - bit)
        return result

    def read_file(self, file_path):
        try:
            # Read the binary data as 32-bit floats
            iq_data = np.fromfile(file_path, dtype=np.float32)

            # Check if file contains complete IQ pairs
            if len(iq_data) % 2 != 0:
                raise ValueError("File does not contain complete IQ pairs")

            # Reshape the data into I and Q components
            i_data = iq_data[0::2]  # Take every other sample starting at 0
            q_data = iq_data[1::2]  # Take every other sample starting at 1

            # Combine into complex signal
            self.sig = i_data + 1j * q_data

            # Convert to GPU if needed
            if self.use_gpu:
                self.sig = self.to_gpu(self.sig)

            return self.to_cpu(self.sig)

        except Exception as e:
            print(f"Error reading file: {str(e)}")
            self.sig = None
            return False

    def write_file(self, file_path, sig=None):
        if sig is None:
            sig = self.sig

        # Make sure we're working with CPU data for file operations
        sig = self.to_cpu(sig)

        i_data = np.real(sig)
        q_data = np.imag(sig)
        iq_interleaved = np.vstack((i_data, q_data)).T.flatten()

        iq_interleaved = iq_interleaved.astype(np.float32)
        iq_interleaved.tofile(file_path)
        return True

    def real_dechirp(self,start = 0,zero_padding = 10, is_up = 1):
        self.zero_padding_ratio = zero_padding
        self.bin_num = 2 ** self.sf * zero_padding
        sig = lora.sig[start:start + lora.get_samples_per_symbol()]
        if (is_up):
            conjugate = lora.ideal_chirp(f0=0,iq_invert=1)
        else:
            conjugate = lora.ideal_chirp(f0=0,iq_invert=0)
        dechirped = sig * conjugate
        ft = np.fft.fft(dechirped,len(dechirped) * zero_padding)
        ft_magnitude = np.abs(ft)
        front = ft_magnitude[:2 ** self.sf * zero_padding]
        end = ft_magnitude[len(ft_magnitude) - 2 ** self.sf * zero_padding:]
        ft_magnitude = front + end
        max_index = np.argmax(ft_magnitude)
        max_value = ft_magnitude[max_index]
        return max_index,max_value
    def ideal_dechirp(self,start = 0,zero_padding = 10, is_up = 1):
        sig = lora.sig[start:start + lora.get_samples_per_symbol()]
        if (is_up):
            conjugate = lora.ideal_chirp(f0=0,iq_invert=1)
        else:
            conjugate = lora.ideal_chirp(f0=0,iq_invert=0)
        dechirped = sig * conjugate
        ft = np.fft.fft(dechirped,len(dechirped) * zero_padding)
        ft_magnitude = np.abs(ft)
        front = ft_magnitude[:2 ** self.sf * zero_padding]
        end = ft_magnitude[len(ft_magnitude) - 2 ** self.sf * zero_padding:]
        ft_magnitude = front + end
        max_index = np.argmax(ft_magnitude)
        max_value = ft_magnitude[max_index]
        return max_index,max_value

    def detect(self, start_index=0):
        ii = start_index
        pk_bin_list = []
        while (ii < len(self.sig) - self.get_samples_per_symbol() * self.preamble_len):
            if (len(pk_bin_list) == self.preamble_len - 1):
                return ii - round((pk_bin_list[-1] ) / self.zero_padding_ratio * self.os_ratio)
            pk0 = self.real_dechirp(ii)
            if (len(pk_bin_list) != 0):
                self.bin_num = self.zero_padding_ratio * 2 ** self.sf
                bin_diff = (pk_bin_list[-1] - pk0[0]) % self.bin_num
                if (bin_diff > self.bin_num / 2):
                    bin_diff = self.bin_num - bin_diff
                if bin_diff <= self.zero_padding_ratio:
                    pk_bin_list.append(pk0[0])
                else:
                    pk_bin_list = [pk0[0]]
            else:
                pk_bin_list = [pk0[0]]
            ii += self.get_samples_per_symbol()
        return -1
    def sync(self,x = 0):
        found = 0
        while(x < len(self.sig) - self.get_samples_per_symbol()):
            up_peak = self.real_dechirp(start=x,zero_padding=self.zero_padding_ratio,is_up=1)
            down_peak = self.real_dechirp(start=x,zero_padding=self.zero_padding_ratio,is_up=0)
            if(up_peak[1] < down_peak[1]):
                found = 1
            x += self.get_samples_per_symbol()
            if (found):
                break
        if (not found):
            return -1
        pkd = self.real_dechirp(x,is_up=0,zero_padding=self.zero_padding_ratio)
        self.fftbin_num = 2 ** self.sf * self.zero_padding_ratio
        if pkd[0] > self.fftbin_num/2:
            coarse_to = round((pkd[0] - self.fftbin_num) / self.zero_padding_ratio)
        else:
            coarse_to = pkd[0] / self.zero_padding_ratio

        x += int(coarse_to) * 4

        pk_d = self.real_dechirp(x,is_up=0,zero_padding=100)
        pk_u = self.real_dechirp(x-4*self.get_samples_per_symbol(),is_up=1,zero_padding=100)
        ud_bin = round((pk_u[0] + pk_d[0])/2)

        if ud_bin > self.bin_num/2:
            self.cfo = (ud_bin - self.bin_num) * self.bw / self.bin_num
        else:
            self.cfo = (ud_bin) * self.bw / self.bin_num
        self.sfo = self.cfo / self.rf_freq * self.bw
        fine_to = round((pk_d[0] - ud_bin) / self.zero_padding_ratio * self.os_ratio * 4)
        x = x + fine_to

        pk_u_last = self.real_dechirp(x - self.get_samples_per_symbol(),is_up=1)
        pk_d_last = self.real_dechirp(x - self.get_samples_per_symbol(), is_up=0)
        if pk_u_last[1] > pk_d_last[1]:
            x_sync = x + round(2.25 * self.get_samples_per_symbol())
        else:
            x_sync = x + round(1.25 * self.get_samples_per_symbol())
        return x_sync

    def comp_offset_td(self):
        self.sfo_accum = self.symbol_cnt * self.sfo
        t = np.arange(self.get_samples_per_symbol()) / self.fs
        phase = 2 * np.pi * (self.sfo_accum + self.cfo) * t
        self.symbol_cnt += 1
        return  np.exp(-1j * phase)
    def limit_demodulate(self,start = 0,symbols = 98, func = None):
        ii = start
        if func == None:
            func = self.our_ideal_decode_decodev2
        self.symbol_cnt = 0
        result = []
        for i in range(symbols):
            sig = lora.sig[ii:ii+self.get_samples_per_symbol()]
            sigc = sig * lora.comp_offset_td()
            result.append(func(sig = sigc))
            ii += self.get_samples_per_symbol()
        return ii
    def limit_save(self, start = 0, num = 64, func = None,prefix = "./ideal",one = 1):
        ii = start
        if func == None:
            func = self.our_ideal_decode_decodev2
        num = num + one
        for i in range(num):
            sig = lora.sig[ii:ii+self.get_samples_per_symbol()]
            sigc = sig * lora.comp_offset_td()
            r = func(sig = sigc)
            print(r)
            lora.write_file(sig = sigc, file_path = prefix +'/'+ str(r[0]) + '.cfile')
            ii += self.get_samples_per_symbol()










if __name__ == '__main__':
    lora = PyLoRa()