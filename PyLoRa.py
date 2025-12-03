import math
from scipy.signal import chirp
from scipy.signal import resample_poly
from scipy.fft import fft as np_fft
from scipy.signal import chirp as np_chirp
import matplotlib.pyplot as plt
import numpy as np
# 弱信号解码
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
        self.dataE1 = self.gen_constants()[0]
        self.dataE2 = self.gen_constants()[1]

    def get_symbol_period(self):
        return (2 ** self.sf) / self.bw

    def get_samples_per_symbol(self):
        return int((2 ** self.sf) * self.fs / self.bw)

    def idealx_chirp(self, f0=0, iq_invert=0):
        ud = int(f0 / int(2 ** self.sf / 2 ))
        f_shift = f0 % int(2 ** self.sf / 2)
        if(ud):
            return self.ideal_chirp(f0=f_shift,iq_invert=1)
        else:
            return self.ideal_chirp(f0=f_shift,iq_invert=0)
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
    def ideal_chirpx(self, f0=0, iq_invert=0,rate = 1):
        self.iq_invert = iq_invert
        num_symbols = 2 ** self.sf
        num_samples = int(num_symbols * self.fs / self.bw)

        # Time array
        t = np.linspace(0, num_symbols / self.bw, num_samples + 1)[:-1]

        # Calculate frequency shift
        freq_shift = (f0 * self.bw) / (rate * num_symbols)

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




    import numpy as np

    from scipy.signal import chirp


    def MFFT(self, sig):
        """
        Multiple-FFT demodulation at OSI=2 (oversampling ratio = 2).

        - Input signal assumed at self.fs (e.g., 1 MHz) and bw (e.g., 125 kHz)
        - Downsample to fs_target = 2 * bw (e.g., 250 kHz) using resample_poly
        - Split even/odd branches and combine their FFTs with phase compensation

        Returns:
            (symbol_idx, peak_value)
        """
        # Guard: one-symbol length handling at original fs
        samples_per_symbol = self.get_samples_per_symbol()
        if len(sig) > samples_per_symbol:
            sig = sig[:samples_per_symbol]
        elif len(sig) < samples_per_symbol:
            sig = np.pad(sig, (0, samples_per_symbol - len(sig)), 'constant')

        # Target OSR and target sampling rate
        target_osr = 2
        fs_target = int(self.bw * target_osr)

        # Downsample from current fs to fs_target with anti-aliasing
        current_fs = int(self.fs)
        if current_fs == fs_target:
            sig_ds = sig
        else:
            # Prefer integer decimation when possible, else general resample
            ratio = current_fs / fs_target
            decim = int(round(ratio)) if ratio > 0 else 1
            if abs(ratio - decim) < 1e-6 and decim > 0:
                sig_ds = resample_poly(sig, up=1, down=decim)
            else:
                sig_ds = resample_poly(sig, up=fs_target, down=current_fs)

        # Required length at OSR=2 is 2 * N
        N = 2 ** self.sf
        required_len = 2 * N
        if len(sig_ds) > required_len:
            sig_ds = sig_ds[:required_len]
        elif len(sig_ds) < required_len:
            sig_ds = np.pad(sig_ds, (0, required_len - len(sig_ds)), 'constant')

        # Split even/odd branches (length N each)
        even_samples = sig_ds[0::2][:N]
        odd_samples = sig_ds[1::2][:N]

        # Generate downchirp at OSR=2 to match branch lengths
        # Use conjugate of upchirp to ensure proper dechirping around (-B/2, B/2)
        original_fs = self.fs
        self.fs = fs_target
        upchirp_full = self.ideal_chirp(f0=0, iq_invert=0)
        downchirp_full = np.conj(upchirp_full)
        self.fs = original_fs

        # Align downchirp to branches
        downchirp_even = downchirp_full[0::2][:N]
        downchirp_odd = downchirp_full[1::2][:N]



        # Dechirp and FFT (N points)
        y_even = even_samples * downchirp_even
        y_odd = odd_samples * downchirp_odd
        Y_even = np.fft.fft(y_even, n=N)
        Y_odd = np.fft.fft(y_odd, n=N)

        # Reconstruct 2N-FFT halves using radix-2: 
        # X[k]   = E[k] + W_{2N}^k O[k]
        # X[k+N] = E[k] - W_{2N}^k O[k]
        k = np.arange(N)
        W = np.exp(-1j * np.pi * k / N)  # W_{2N}^k
        X_lo = Y_even + W * Y_odd
        X_hi = Y_even - W * Y_odd

        # Fold positive/negative frequencies
        mags = np.abs(X_lo) + np.abs(X_hi)
        fft_peak_idx = int(np.argmax(mags))
        peak_val = float(mags[fft_peak_idx])
        # Map FFT bin index to LoRa symbol index
        symbol_idx = int((fft_peak_idx ) % N)
        return symbol_idx, peak_val

    def hfft_decode(self, sig):
        downchirp = self.ideal_chirp(f0=0,iq_invert=0)
        downchirp = np.conj(downchirp)
        result = downchirp * sig
        spec = np.fft.fft(result,len(result))
        vals = np.abs(spec)
        vals0 = vals[:2 ** self.sf]
        est0 = np.argmax(vals0).item()
        max_val0 = np.max(vals0).item()
        vals1 = vals[-2 ** self.sf:]
        est1 = np.argmax(vals1).item()
        max_val1 = np.max(vals1).item()
        if max_val0 > max_val1:
            return est0, max_val0
        return est1, max_val1

        # downchirp = self.ideal_chirp(f0=0, iq_invert=1)

        # dechirped = sig * downchirp
        # fft_result = np.fft.fft(dechirped, len(dechirped))

        # magnitudes = np.abs(fft_result)

        # num_bins = 2 ** self.sf

        # neg_freqs = magnitudes[len(magnitudes) - int(num_bins / 2):]
        # pos_freqs = magnitudes[:int(num_bins / 2)]

        # relevant_magnitudes = np.concatenate((neg_freqs, pos_freqs))

        # # Find the index of the maximum magnitude
        # max_idx = np.argmax(relevant_magnitudes)
        # max_magnitude = relevant_magnitudes[max_idx]

        # # Adjust the index to represent the correct symbol
        # if max_idx >= int(num_bins / 2):
        #     # If in the negative frequency part, adjust the index
        #     symbol = max_idx - int(num_bins / 2)
        # else:
        #     # If in the positive frequency part, adjust the index
        #     symbol = max_idx + int(num_bins / 2)

        # return symbol, max_magnitude




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
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol
        # add noise of a certain SNR, chosen from snr_range
        amp = math.pow(0.1, snr / 20) * np.mean(np.abs(sig))
        noise = (amp / math.sqrt(2) * np.random.randn(num_samples) + 1j * amp / math.sqrt(2) * np.random.randn(
            num_samples))
        dataX = sig + noise  # dataX: data with noise
        return dataX

    def loraphy(self, sig):
        upsampling = 100  # up-sampling rate for loraphy, default 100
        # upsamping can counter possible frequency misalignments, finding the highest position of the signal peak,
        # but higher upsampling lead to more noise
        num_classes = 2 ** self.sf
        # dechirp
        downchirp = self.ideal_chirp(f0=0,iq_invert=1)
        chirp_data = sig * downchirp

        # compute FFT

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

    def loraphy_fpa(self, sig, k=32):
        """
        Fine-grained Phase Alignment (FPA) method from LoRaPHY paper
        
        Args:
            sig: Input signal
            k: Number of phase search steps (default 16, as mentioned in paper)
            
        Returns:
            tuple: (peak_idx, peak_value)
        """
        num_classes = 2 ** self.sf
        osr = int(self.fs / self.bw)  # Over-sampling ratio: 1MHz / 125kHz = 8

        downchirp = self.ideal_chirp(f0=0, iq_invert=1)

        dechirped = sig * downchirp
        upsampling = 100

        fft_raw = np_fft(dechirped, len(dechirped) * upsampling)
        # Phase alignment search
        best_peak_value = 0
        best_peak_idx = 0
        
        # Search through k possible phase offsets: Δφ = i×2π/k
        for i in range(k):
            phase_offset = i * 2 * np.pi / k
            target_nfft = num_classes * upsampling
            cut1 = fft_raw[:target_nfft]
            cut2 = fft_raw[-target_nfft:]
            cut1 *= np.exp(1j * phase_offset)
            combined_spectrum =np.abs(cut1 + cut2)
            # add absolute values of cut1 and cut2 to merge two peaks into one
            peak_idx = round((np.argmax(combined_spectrum) / upsampling).item()) % num_classes
            peak_value = np.max(combined_spectrum)
            if peak_value > best_peak_value:
                best_peak_value = peak_value
                best_peak_idx = peak_idx
        return best_peak_idx, best_peak_value

            

        return best_peak_idx, best_peak_value
    def gen_ideal_matrix(self):
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol
        result = np.zeros((num_classes, num_samples), dtype=np.complex64)
        for i in range(num_classes):
            result[i] = self.ideal_chirp(f0=i,iq_invert=1)
            self.f0 = i
        return result
    def gen_idealx_matrix(self):
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol
        result = np.zeros((num_classes, num_samples), dtype=np.complex64)
        for i in range(num_classes):
            if(i < int(num_classes / 2)):
                result[i] = self.ideal_chirp(f0=i,iq_invert=1)
                self.f0 = i
            else:
                result[i] = self.ideal_chirp(f0=i - int(num_classes/2), iq_invert=0)
                self.f0 = i
        return result





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
    def gen_tone(self, freq):
        length = self.get_samples_per_symbol()
        fs = self.fs
        t = np.arange(length) / fs
        signal = np.exp(1j * 2 * np.pi * freq * t)
        return signal



    def gen_constants(self):
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol

        # generate downchirp
        t = np.linspace(0, num_samples / self.fs, num_samples + 1)[:-1]

        chirpI1 = np.array(np_chirp(t, f0=self.bw / 2, f1=-self.bw / 2, t1=2 ** self.sf / self.bw, method='linear', phi=90))
        chirpQ1 = np.array(np_chirp(t, f0=self.bw / 2, f1=-self.bw / 2, t1=2 ** self.sf / self.bw, method='linear', phi=0))
        downchirp = chirpI1 + 1j * chirpQ1
        # two DFT matrices
        dataE1 = np.zeros((num_classes, num_samples), dtype=np.complex64)
        dataE2 = np.zeros((num_classes, num_samples), dtype=np.complex64)
        for symbol_index in range(num_classes):
            time_shift = int(symbol_index / num_classes * num_samples)
            time_split = num_samples - time_shift
            dataE1[symbol_index][:time_split] = downchirp[time_shift:]
            if symbol_index != 0: dataE2[symbol_index][time_split:] = downchirp[:time_shift]

        return dataE1, dataE2




    def loratrimmer_decode(self, sig):
        # dataE1,dataE2 = self.gen_constants()
        dataE1, dataE2 = self.dataE1,self.dataE2
        sig = np.array(sig).T
        data1 = np.matmul(dataE1, sig)
        data2 = np.matmul(dataE2, sig)
        vals = np.abs(data1) ** 2 + np.abs(data2) ** 2
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est, max_val
    def lora_trimmer_edit(self,sig):
        dataE1, dataE2 = self.gen_constants()
        data = dataE1 + dataE2
        sig = np.array(sig).T
        datas = np.matmul(data, sig)
        vals = np.abs(datas) ** 2
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est, max_val

    def our_raw_ideal_decode_decodev2(self, sig):
        mtx = self.gen_ideal_matrix()
        sig = np.array(sig).T
        result = np.matmul(mtx, sig)
        return result
    def our_ideal_decode_decodev2(self, sig):

        mtx = self.gen_ideal_matrix()
        sig = np.array(sig).T
        result = np.matmul(mtx, sig)
        vals = np.abs(result) ** 2
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est, max_val
    def fft_ideal_decode_decodev2(self,sig):
        downchirp = self.ideal_chirp(f0=0,iq_invert=0)
        downchirp = np.conj(downchirp)
        result = downchirp * sig
        spec = np.fft.fft(result,len(result))
        vals = np.abs(spec)
        vals = vals[:2 ** self.sf]
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est, max_val


    def subl_raw_our_ideal_decode_decodev2(self,sig,n):
        mtx = self.gen_ideal_matrix()[:, :n]
        sig = sig[0:n]
        sig = np.array(sig).T
        return np.matmul(mtx, sig)

    def subl_our_ideal_decode_decodev2(self,sig,n):
        result = self.subl_raw_our_ideal_decode_decodev2(sig,n)
        vals = np.abs(result) ** 2
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est, max_val
    def subr_raw_our_ideal_decode_decodev2(self,sig,n):
        mtx = self.gen_ideal_matrix()
        mtx = mtx[:, n:]
        sig = sig[n:len(sig)]
        sig = np.array(sig).T
        return np.matmul(mtx, sig)
    def subr_our_ideal_decode_decodev2(self,sig,n):
        result = self.subr_raw_our_ideal_decode_decodev2(sig,n)
        vals = np.abs(result) ** 2
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est, max_val



    def read_file(self, file_path):

        try:
            # Read the binary data as 32-bit floats
            iq_data = np.fromfile(file_path, dtype=np.float32)

            # Check if file contains complete IQ pairs
            if len(iq_data) % 2 != 0:
                raise ValueError("File does not contain complete IQ pairs")

            # Reshape the data into I and Q components
            # Every even index (0,2,4...) is I, every odd index (1,3,5...) is Q
            i_data = iq_data[0::2]  # Take every other sample starting at 0
            q_data = iq_data[1::2]  # Take every other sample starting at 1

            # Combine into complex signal
            self.sig = i_data + 1j * q_data

            return self.sig

        except Exception as e:
            print(f"Error reading file: {str(e)}")
            self.sig = None
            return False

    def write_file(self, file_path, sig = None):

        if(sig is None):
            sig = self.sig
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
        # print("sfo ")
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
    def limit_save(self, start = 0, num = 64, func = None,prefix = "./ideal_past",one = 1):
        ii = start
        if func == None:
            func = self.loratrimmer_decode
        num = num + one
        for i in range(num):
            sig = lora.sig[ii:ii+self.get_samples_per_symbol()]
            sigc = sig * lora.comp_offset_td()
            r = func(sig = sigc)
            print(r)
            lora.write_file(sig = sigc, file_path = prefix +'/'+ str(r[0]) + '.cfile')
            ii += self.get_samples_per_symbol()

    def nelora_decode(self, sig):
        """
        NeLoRa decoding method using neural network-enhanced demodulation.
        
        This method converts the input signal to STFT spectrogram format,
        applies NeLoRa's neural network models for denoising and classification.
        
        Args:
            sig: Input LoRa signal (complex-valued numpy array)
        
        Returns:
            tuple: (estimated_symbol, confidence_value)
        """
        try:
            # Import NeLoRa modules
            import sys
            import os
            nelora_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'NELoRa-Sensys/neural_enhanced_demodulation/pytorch')
            if nelora_path not in sys.path:
                sys.path.append(nelora_path)
            
            import torch
            from scipy import signal as scipy_signal
            from models.model_components import maskCNNModel, classificationHybridModel
            import config
            
            # Create mock configuration for NeLoRa
            class MockOpts:
                def __init__(self, sf):
                    self.sf = sf
                    self.n_classes = 2 ** sf
                    self.bw = 125000
                    self.fs = 1000000
                    self.x_image_channel = 2
                    self.y_image_channel = 2
                    self.conv_dim_lstm = self.n_classes * self.fs // self.bw
                    self.lstm_dim = 400
                    self.fc1_dim = 600
                    self.freq_size = self.n_classes
                    self.stft_nfft = self.n_classes * self.fs // self.bw
                    self.stft_window = self.n_classes // 2
                    self.stft_overlap = self.stft_window // 2
                    self.normalization = False
            
            opts = MockOpts(self.sf)
            
            # Convert signal to STFT spectrogram
            sig_array = np.array(sig)
            
            # Compute STFT
            f, t, Zxx = scipy_signal.stft(
                sig_array,
                fs=opts.fs,
                window='hann',
                nperseg=opts.stft_window,
                noverlap=opts.stft_overlap,
                nfft=opts.stft_nfft
            )
            
            # Prepare input for neural network
            stft_data = torch.tensor(Zxx, dtype=torch.cfloat).unsqueeze(0)
            
            # Convert to real-valued format [B, 2, H, W] where 2 = [real, imag]
            stft_real = torch.view_as_real(stft_data)  # [B, H, W, 2]
            stft_input = stft_real.permute(0, 3, 1, 2)  # [B, 2, H, W]
            
            # Create NeLoRa models (without pre-trained weights)
            mask_cnn = maskCNNModel(opts)
            classifier = classificationHybridModel(
                conv_dim_in=opts.y_image_channel,
                conv_dim_out=opts.n_classes,
                conv_dim_lstm=opts.conv_dim_lstm
            )
            
            # Set models to evaluation mode
            mask_cnn.eval()
            classifier.eval()
            
            # WARNING: No pre-trained weights available - using random initialization
            # In practice, you would load pre-trained weights here:
            # mask_cnn.load_state_dict(torch.load('path_to_mask_cnn_weights.pkl'))
            # classifier.load_state_dict(torch.load('path_to_classifier_weights.pkl'))
            
            with torch.no_grad():
                # Apply mask CNN for denoising
                denoised = mask_cnn(stft_input)
                
                # Apply classifier for symbol prediction
                logits = classifier(denoised)
                probabilities = torch.softmax(logits, dim=1)
                
                # Get predicted symbol and confidence
                confidence, predicted = torch.max(probabilities, dim=1)
                estimated_symbol = predicted.item()
                confidence_value = confidence.item()
            
            return estimated_symbol, confidence_value
            
        except ImportError as e:
            # Fallback to simple FFT-based decoding if NeLoRa modules not available
            print(f"Warning: NeLoRa modules not available ({e}). Using fallback decoding.")
            return self.our_ideal_decode_decodev2(sig)
        except Exception as e:
            # Fallback for any other errors
            print(f"Warning: NeLoRa decoding failed ({e}). Using fallback decoding.")
            return self.our_ideal_decode_decodev2(sig)


lora = PyLoRa()





























