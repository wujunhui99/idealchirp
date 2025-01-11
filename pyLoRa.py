import math
from scipy.signal import chirp
from scipy.fft import fft as np_fft
from scipy.signal import chirp as np_chirp
import matplotlib.pyplot as plt
import numpy as np
import unittest
class pyLoRa:
    def __init__(self, sf=7, bw=125e3,iq_invert=0 ,fs=1e6,sig = None, zero_padding = 10,payload=None, f0=0, preamble_len=6,raw_chirp=None,rf_freq = 915e6):
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
    def our_ideal_decode(self, ideal_data):
        down_chirp = self.ideal_chirp(f0=0, iq_invert=1)
        dechirp = ideal_data * down_chirp
        # Compute FFT
        fft_result = np.fft.fft(dechirp)

        # Get magnitude spectrum (absolute values)
        magnitudes = np.abs(fft_result)



        max_magnitude = np.max(magnitudes)
        max_idx = np.argmax(magnitudes)
        return max_idx, max_magnitude
    def loraphy_CPA(self, real_data):
        down_chirp = self.ideal_chirp(f0=0, iq_invert=1)
        dechirp = real_data * down_chirp
        # Compute FFT
        fft_result = np.fft.fft(dechirp)

        # Get magnitude spectrum (absolute values)
        magnitudes = np.abs(fft_result)


        n = len(magnitudes) // 2
        magnitudes = magnitudes[:n] + np.append(np.flip( magnitudes[n:] ), magnitudes[n])[1:]
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
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol
        # add noise of a certain SNR, chosen from snr_range
        amp = math.pow(0.1, snr / 20) * np.mean(np.abs(sig))
        noise = (amp / math.sqrt(2) * np.random.randn(num_samples) + 1j * amp / math.sqrt(2) * np.random.randn(
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

    def loraphy_FPA(self, data_in):
        upsampling = 100  # up-sampling rate for loraphy, default 100
        # upsamping can counter possible frequency misalignments, finding the highest position of the signal peak,
        # but higher upsampling lead to more noise
        num_classes = 2 ** self.sf
        # dechirp
        downchirp = self.ideal_chirp(f0=0,iq_invert=1)
        chirp_data = data_in * downchirp

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
    def gen_ideal_matrix(self):
        num_classes = 2 ** self.sf  # number of codes per symbol == 2 ** sf
        num_samples = int(num_classes * self.fs / self.bw)  # number of samples per symbol
        result = np.zeros((num_classes, num_samples), dtype=np.complex64)
        for i in range(num_classes):
            result[i] = self.ideal_chirp(f0=i,iq_invert=1)
            self.f0 = i
        return result
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
        dataE1,dataE2 = self.gen_constants()
        sig = np.array(sig).T
        data1 = np.matmul(dataE1, sig)
        data2 = np.matmul(dataE2, sig)
        vals = np.abs(data1) ** 2 + np.abs(data2) ** 2
        est = np.argmax(vals).item()
        max_val = np.max(vals).item()
        return est, max_val
    def our_ideal_decode_decodev2(self, sig):
        mtx = self.gen_ideal_matrix()
        sig = np.array(sig).T
        result = np.matmul(mtx, sig)
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

            return True

        except Exception as e:
            print(f"Error reading file: {str(e)}")
            self.sig = None
            return False

    def write_file(self, file_path):
        try:
            if self.sig is None:
                raise ValueError("No signal data to write (self.sig is None)")

            i_data = np.real(self.sig)
            q_data = np.imag(self.sig)
            iq_interleaved = np.vstack((i_data, q_data)).T.flatten()

            iq_interleaved = iq_interleaved.astype(np.float32)
            iq_interleaved.tofile(file_path)
            return True
        except Exception as e:
            print(f"Error writing file: {str(e)}")
            return False
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
                return ii - round((pk_bin_list[-1] ) / self.zero_padding_ratio * 8)
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
        """
        补偿CFO和累积的SFO
        """
        # 计算累积的SFO偏移
        self.sfo_accum = self.symbol_cnt * self.sfo

        # 生成时间序列
        t = np.arange(self.sample_num) / self.fs

        # 计算补偿信号
        phase = 2 * np.pi * (self.sfo_accum + self.cfo) * t
        sfo_comp_sig = np.exp(1j * phase)

        # 对基准downchirp进行补偿
        self.downchirp = self.basedownchirp * sfo_comp_sig


    '''
    
function x_sync = sync(self, x)
   


% Up-Down Alignment: downchirp和upchirp的峰值对齐
    % 窗口完全对齐      窗口向左偏：pku减小，pkd增大      窗口向右偏：pku增大，pkd减小
    % [  /|\  ]              [   |/\ ]                     [ /\|   ]
    % [ / | \ ]              [  /|  \]                     [/  |\  ]
    % [/  |  \]              [ / |   ]\                   /[   | \ ]



% 细粒度窗口对齐: 消除sto，对齐到payload的起点x_sync
    fine_to = round((pk_d(1,1)-ud_bin)/self.zero_padding_ratio*self.os_ratio);
    x = x + fine_to;

    pk_u_last = self.dechirp(x-self.sample_num, true);
    pk_d_last = self.dechirp(x-self.sample_num, false);
    if abs(pk_u_last(1,2)) > abs(pk_d_last(1,2))
        % last chirp is upchirp, so current symbol is the first downchirp
        x_sync = x + round(2.25*self.sample_num);
    else
        % last chirp is downchirp, so current symbol is the second downchirp
        x_sync = x + round(1.25*self.sample_num);
    end

    % 对齐成功，计数解调符号
    self.symbol_cnt = 0;
end % sync
    '''



# 1419592

lora = pyLoRa()
if __name__ == '__main__':

    print(lora.read_file("/Users/junhui/code/test/up_upchirp.cfile"))
    lora.preamble_len = 8
    x = lora.detect(start_index=0)
    print(x)
    print(lora.sync(x=x))



    pass









