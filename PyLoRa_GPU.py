import math
import numpy as np
try:
    import cupy as cp
    from cupy import fft as cp_fft
except Exception as e:
    raise RuntimeError(f"CuPy is required for PyLoRa_GPU but failed to import: {e}")


class PyLoRa:
    def __init__(self, sf=7, bw=125e3, iq_invert=0, fs=1e6, sig=None, zero_padding=10, payload=None, f0=0, preamble_len=6, raw_chirp=None, rf_freq=915e6, bit=5):
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

    # --------- helpers (GPU) ---------
    def _to_gpu(self, x, dtype=cp.complex64):
        if isinstance(x, cp.ndarray):
            return x.astype(dtype, copy=False)
        return cp.asarray(x, dtype=dtype)

    def _ideal_chirp_gpu(self, f0=0, iq_invert=0, fs=None, rate=1):
        fs = self.fs if fs is None else fs
        num_symbols = 2 ** self.sf
        num_samples = int(num_symbols * fs / self.bw)
        t = cp.linspace(0, num_symbols / self.bw, num_samples + 1, dtype=cp.float64)[:-1]
        freq_shift = (f0 * self.bw) / (rate * num_symbols)
        f0_shifted = -self.bw / 2 + freq_shift
        f1_shifted = self.bw / 2 + freq_shift
        T = num_symbols / self.bw
        k = (f1_shifted - f0_shifted) / T
        phase = 2 * cp.pi * (f0_shifted * t + 0.5 * k * t * t)
        # Match SciPy chirp usage exactly: I = cos(phase + 90deg), Q = cos(phase + 0deg)
        chirp_i = cp.cos(phase + cp.pi / 2)
        chirp_q = cp.cos(phase)
        if iq_invert:
            chirp_q = -chirp_q
        signal = chirp_i + 1j * chirp_q
        return signal.astype(cp.complex64)

    # --------- basic getters ---------
    def get_symbol_period(self):
        return (2 ** self.sf) / self.bw

    def get_samples_per_symbol(self):
        return int((2 ** self.sf) * self.fs / self.bw)

    # --------- GPU-accelerated decoders used by tests ---------
    def MFFT(self, sig):
        samples_per_symbol = self.get_samples_per_symbol()
        if len(sig) > samples_per_symbol:
            sig = sig[:samples_per_symbol]
        elif len(sig) < samples_per_symbol:
            sig = np.pad(sig, (0, samples_per_symbol - len(sig)), 'constant')

        sig_gpu = self._to_gpu(sig)

        target_osr = 2
        fs_target = int(self.bw * target_osr)
        current_fs = int(self.fs)
        # Use high-quality polyphase resampling to avoid aliasing
        if current_fs == fs_target:
            sig_ds = sig_gpu
        else:
            from scipy.signal import resample_poly
            sig_cpu = cp.asnumpy(sig_gpu)
            sig_cpu = resample_poly(sig_cpu, up=fs_target, down=current_fs)
            sig_ds = self._to_gpu(sig_cpu)

        N = 2 ** self.sf
        required_len = 2 * N
        if sig_ds.shape[0] > required_len:
            sig_ds = sig_ds[:required_len]
        elif sig_ds.shape[0] < required_len:
            pad = required_len - sig_ds.shape[0]
            sig_ds = cp.pad(sig_ds, (0, pad))

        even_samples = sig_ds[0::2][:N]
        odd_samples = sig_ds[1::2][:N]

        downchirp_full = cp.conj(self._ideal_chirp_gpu(f0=0, iq_invert=0, fs=fs_target))
        downchirp_even = downchirp_full[0::2][:N]
        downchirp_odd = downchirp_full[1::2][:N]

        y_even = even_samples * downchirp_even
        y_odd = odd_samples * downchirp_odd
        Y_even = cp_fft.fft(y_even, n=N)
        Y_odd = cp_fft.fft(y_odd, n=N)

        k = cp.arange(N)
        W = cp.exp(-1j * cp.pi * k / N)
        X_lo = Y_even + W * Y_odd
        X_hi = Y_even - W * Y_odd

        mags = cp.abs(X_lo) + cp.abs(X_hi)
        fft_peak_idx = int(cp.argmax(mags).get())
        peak_val = float(cp.max(mags).get())
        symbol_idx = int((fft_peak_idx) % N)
        return symbol_idx, peak_val

    def hfft_decode(self, sig):
        sig_gpu = self._to_gpu(sig)
        downchirp = self._ideal_chirp_gpu(f0=0, iq_invert=1)
        dechirped = sig_gpu * downchirp
        fft_result = cp_fft.fft(dechirped, dechirped.shape[0])
        magnitudes = cp.abs(fft_result)
        num_bins = 2 ** self.sf
        neg_freqs = magnitudes[magnitudes.shape[0] - int(num_bins / 2):]
        pos_freqs = magnitudes[:int(num_bins / 2)]
        relevant_magnitudes = cp.concatenate((neg_freqs, pos_freqs))
        max_idx = int(cp.argmax(relevant_magnitudes).get())
        max_magnitude = float(cp.max(relevant_magnitudes).get())
        if max_idx >= int(num_bins / 2):
            symbol = max_idx - int(num_bins / 2)
        else:
            symbol = max_idx + int(num_bins / 2)
        return symbol, max_magnitude

    def loraphy(self, sig):
        upsampling = 256
        num_classes = 2 ** self.sf
        downchirp = self._ideal_chirp_gpu(f0=0, iq_invert=1)
        sig_gpu = self._to_gpu(sig)
        dechirped = sig_gpu * downchirp
        fft_raw = cp_fft.fft(dechirped, dechirped.shape[0] * upsampling)
        target_nfft = num_classes * upsampling
        cut1 = fft_raw[:target_nfft]
        cut2 = fft_raw[-target_nfft:]
        combined_spectrum = cp.abs(cut1) + cp.abs(cut2)
        peak_idx = int(round((int(cp.argmax(combined_spectrum).get()) / upsampling)) % num_classes)
        peak_value = float(cp.max(combined_spectrum).get())
        return peak_idx, peak_value

    def loraphy_fpa(self, sig, k=32):
        num_classes = 2 ** self.sf
        downchirp = self._ideal_chirp_gpu(f0=0, iq_invert=1)
        sig_gpu = self._to_gpu(sig)
        dechirped = sig_gpu * downchirp
        upsampling = 256
        fft_raw = cp_fft.fft(dechirped, dechirped.shape[0] * upsampling)
        best_peak_value = 0.0
        best_peak_idx = 0
        target_nfft = num_classes * upsampling
        cut1_base = fft_raw[:target_nfft]
        cut2 = fft_raw[-target_nfft:]
        for i in range(k):
            phase_offset = i * 2 * math.pi / k
            cut1 = cut1_base * cp.exp(1j * phase_offset)
            combined_spectrum = cp.abs(cut1 + cut2)
            peak_idx = int(round((int(cp.argmax(combined_spectrum).get()) / upsampling)) % num_classes)
            peak_value = float(cp.max(combined_spectrum).get())
            if peak_value > best_peak_value:
                best_peak_value = peak_value
                best_peak_idx = peak_idx
        return best_peak_idx, best_peak_value

    # --------- Batched GPU decoders to increase utilization ---------
    def batch_hfft_decode(self, sigs):
        sig_gpu = self._to_gpu(sigs)
        downchirp = self._ideal_chirp_gpu(f0=0, iq_invert=1)
        dechirped = sig_gpu * downchirp  # broadcast over batch
        fft_result = cp_fft.fft(dechirped, dechirped.shape[-1], axis=-1)
        magnitudes = cp.abs(fft_result)
        num_bins = 2 ** self.sf
        neg_freqs = magnitudes[..., magnitudes.shape[-1] - int(num_bins / 2):]
        pos_freqs = magnitudes[..., :int(num_bins / 2)]
        relevant_magnitudes = cp.concatenate((neg_freqs, pos_freqs), axis=-1)
        max_idx = cp.argmax(relevant_magnitudes, axis=-1)
        # map to symbol per batch
        half = int(num_bins / 2)
        sym = cp.where(max_idx >= half, max_idx - half, max_idx + half)
        max_mag = cp.take_along_axis(relevant_magnitudes, max_idx[..., None], axis=-1).squeeze(-1)
        return cp.asnumpy(sym).astype(int), cp.asnumpy(max_mag).astype(float)

    def batch_loraphy(self, sigs):
        upsampling = 256
        num_classes = 2 ** self.sf
        sig_gpu = self._to_gpu(sigs)
        downchirp = self._ideal_chirp_gpu(f0=0, iq_invert=1)
        dechirped = sig_gpu * downchirp
        nfft = dechirped.shape[-1] * upsampling
        fft_raw = cp_fft.fft(dechirped, nfft, axis=-1)
        target_nfft = num_classes * upsampling
        cut1 = fft_raw[..., :target_nfft]
        cut2 = fft_raw[..., -target_nfft:]
        combined = cp.abs(cut1) + cp.abs(cut2)
        arg = cp.argmax(combined, axis=-1)
        peak_idx = (cp.rint(arg / upsampling).astype(int)) % num_classes
        peak_val = cp.take_along_axis(combined, arg[..., None], axis=-1).squeeze(-1)
        return cp.asnumpy(peak_idx).astype(int), cp.asnumpy(peak_val).astype(float)

    def batch_loraphy_fpa(self, sigs, k=32):
        num_classes = 2 ** self.sf
        upsampling = 256
        sig_gpu = self._to_gpu(sigs)
        downchirp = self._ideal_chirp_gpu(f0=0, iq_invert=1)
        dechirped = sig_gpu * downchirp
        nfft = dechirped.shape[-1] * upsampling
        fft_raw = cp_fft.fft(dechirped, nfft, axis=-1)
        target_nfft = num_classes * upsampling
        cut1_base = fft_raw[..., :target_nfft]
        cut2 = fft_raw[..., -target_nfft:]
        best_val = cp.zeros(cut1_base.shape[:-1], dtype=cp.float32)
        best_idx = cp.zeros(cut1_base.shape[:-1], dtype=cp.int32)
        for i in range(k):
            phase_offset = i * 2 * math.pi / k
            cut1 = cut1_base * cp.exp(1j * phase_offset)
            combined = cp.abs(cut1 + cut2)
            arg = cp.argmax(combined, axis=-1)
            val = cp.take_along_axis(combined, arg[..., None], axis=-1).squeeze(-1)
            upd = val > best_val
            best_val = cp.where(upd, val, best_val)
            candidate_idx = (cp.rint(arg / upsampling).astype(cp.int32)) % num_classes
            best_idx = cp.where(upd, candidate_idx, best_idx)
        return cp.asnumpy(best_idx).astype(int), cp.asnumpy(best_val).astype(float)

    def batch_our_ideal_decode_decodev2(self, sigs):
        # sigs: (B, T)
        mtx = self.gen_ideal_matrix()  # (N, T)
        sig_gpu = self._to_gpu(sigs)  # (B, T)
        # (N, T) @ (T, B) -> (N, B)
        result = mtx @ sig_gpu.T
        vals = cp.abs(result) ** 2
        est = cp.argmax(vals, axis=0)
        max_val = cp.max(vals, axis=0)
        return cp.asnumpy(est).astype(int), cp.asnumpy(max_val).astype(float)

    def batch_loratrimmer_decode(self, sigs):
        dataE1, dataE2 = self.gen_constants()  # (N,T)
        sig_gpu = self._to_gpu(sigs).T  # (T,B)
        data1 = dataE1 @ sig_gpu  # (N,B)
        data2 = dataE2 @ sig_gpu  # (N,B)
        vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
        est = cp.argmax(vals, axis=0)
        max_val = cp.max(vals, axis=0)
        return cp.asnumpy(est).astype(int), cp.asnumpy(max_val).astype(float)

    def batch_MFFT(self, sigs):
        # Batched version of MFFT with OSR=2; may fall back to CPU resample if needed
        B, T = sigs.shape
        samples_per_symbol = self.get_samples_per_symbol()
        if T != samples_per_symbol:
            # pad/trim per batch
            if T > samples_per_symbol:
                sigs = sigs[:, :samples_per_symbol]
            else:
                import numpy as _np
                pad = samples_per_symbol - T
                sigs = _np.pad(sigs, ((0,0),(0,pad)))
        sig_gpu = self._to_gpu(sigs)
        target_osr = 2
        fs_target = int(self.bw * target_osr)
        current_fs = int(self.fs)
        # Use high-quality polyphase resampling to avoid aliasing
        if current_fs == fs_target:
            sig_ds = sig_gpu
        else:
            from scipy.signal import resample_poly
            sig_cpu = cp.asnumpy(sig_gpu)
            sig_cpu = resample_poly(sig_cpu, up=fs_target, down=current_fs, axis=-1)
            sig_ds = self._to_gpu(sig_cpu)
        N = 2 ** self.sf
        required_len = 2 * N
        if sig_ds.shape[-1] != required_len:
            if sig_ds.shape[-1] > required_len:
                sig_ds = sig_ds[..., :required_len]
            else:
                sig_ds = cp.pad(sig_ds, ((0,0),(0, required_len - sig_ds.shape[-1])))
        even_samples = sig_ds[:, 0::2][:, :N]
        odd_samples = sig_ds[:, 1::2][:, :N]
        downchirp_full = cp.conj(self._ideal_chirp_gpu(f0=0, iq_invert=0, fs=fs_target))
        downchirp_even = downchirp_full[0::2][:N]
        downchirp_odd = downchirp_full[1::2][:N]
        y_even = even_samples * downchirp_even
        y_odd = odd_samples * downchirp_odd
        Y_even = cp_fft.fft(y_even, n=N, axis=-1)
        Y_odd = cp_fft.fft(y_odd, n=N, axis=-1)
        k = cp.arange(N)
        W = cp.exp(-1j * cp.pi * k / N)
        X_lo = Y_even + W * Y_odd
        X_hi = Y_even - W * Y_odd
        mags = cp.abs(X_lo) + cp.abs(X_hi)
        fft_peak_idx = cp.argmax(mags, axis=-1)
        peak_val = cp.max(mags, axis=-1)
        symbol_idx = (fft_peak_idx % N).astype(cp.int32)
        return cp.asnumpy(symbol_idx).astype(int), cp.asnumpy(peak_val).astype(float)

    def add_noise_batch(self, sigs, snr):
        # sigs: numpy or cupy (B,T), returns numpy complex with noise added (on GPU for speed)
        sig_gpu = self._to_gpu(sigs)
        num_classes = 2 ** self.sf
        num_samples = int(num_classes * self.fs / self.bw)
        amp = math.pow(0.1, snr / 20) * float(cp.mean(cp.abs(sig_gpu)))
        noise = (amp / math.sqrt(2) * (cp.random.randn(*sig_gpu.shape) + 1j * cp.random.randn(*sig_gpu.shape))).astype(cp.complex64)
        out = sig_gpu + noise
        return cp.asnumpy(out)

    def gen_ideal_matrix(self):
        num_classes = 2 ** self.sf
        num_samples = int(num_classes * self.fs / self.bw)
        result = cp.zeros((num_classes, num_samples), dtype=cp.complex64)
        for i in range(num_classes):
            result[i] = self._ideal_chirp_gpu(f0=i, iq_invert=1)
            self.f0 = i
        return result

    def our_raw_ideal_decode_decodev2(self, sig):
        mtx = self.gen_ideal_matrix()
        sig_gpu = self._to_gpu(sig).T
        result = mtx @ sig_gpu
        return cp.asnumpy(result)

    def our_ideal_decode_decodev2(self, sig):
        mtx = self.gen_ideal_matrix()
        sig_gpu = self._to_gpu(sig).T
        result = mtx @ sig_gpu
        vals = cp.abs(result) ** 2
        est = int(cp.argmax(vals).get())
        max_val = float(cp.max(vals).get())
        return est, max_val

    def gen_constants(self):
        num_classes = 2 ** self.sf
        num_samples = int(num_classes * self.fs / self.bw)
        # generate downchirp via analytic formula using same convention as CPU
        t = cp.linspace(0, num_samples / self.fs, num_samples + 1, dtype=cp.float64)[:-1]
        f0_lin = self.bw / 2
        f1_lin = -self.bw / 2
        T = (2 ** self.sf) / self.bw
        k = (f1_lin - f0_lin) / T
        phase = 2 * cp.pi * (f0_lin * t + 0.5 * k * t * t)
        chirp_i = cp.cos(phase + cp.pi / 2)
        chirp_q = cp.cos(phase)
        downchirp = (chirp_i + 1j * chirp_q).astype(cp.complex64)
        dataE1 = cp.zeros((num_classes, num_samples), dtype=cp.complex64)
        dataE2 = cp.zeros((num_classes, num_samples), dtype=cp.complex64)
        for symbol_index in range(num_classes):
            time_shift = int(symbol_index / num_classes * num_samples)
            time_split = num_samples - time_shift
            dataE1[symbol_index][:time_split] = downchirp[time_shift:]
            if symbol_index != 0:
                dataE2[symbol_index][time_split:] = downchirp[:time_shift]
        return dataE1, dataE2

    def loratrimmer_decode(self, sig):
        dataE1, dataE2 = self.gen_constants()
        sig_gpu = self._to_gpu(sig).T
        data1 = dataE1 @ sig_gpu
        data2 = dataE2 @ sig_gpu
        vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
        est = int(cp.argmax(vals).get())
        max_val = float(cp.max(vals).get())
        return est, max_val

    # --------- utilities, I/O, and misc (copied behavior) ---------
    def read_file(self, file_path):
        try:
            iq_data = np.fromfile(file_path, dtype=np.float32)
            if len(iq_data) % 2 != 0:
                raise ValueError("File does not contain complete IQ pairs")
            i_data = iq_data[0::2]
            q_data = iq_data[1::2]
            self.sig = i_data + 1j * q_data
            return self.sig
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            self.sig = None
            return False

    def write_file(self, file_path, sig=None):
        if sig is None:
            sig = self.sig
        i_data = np.real(sig)
        q_data = np.imag(sig)
        iq_interleaved = np.vstack((i_data, q_data)).T.flatten().astype(np.float32)
        iq_interleaved.tofile(file_path)
        return True

    def add_noise(self, sig, snr):
        num_classes = 2 ** self.sf
        num_samples = int(num_classes * self.fs / self.bw)
        amp = math.pow(0.1, snr / 20) * np.mean(np.abs(sig))
        # CPU noise then move to GPU when used
        noise = (amp / math.sqrt(2) * np.random.randn(num_samples) + 1j * amp / math.sqrt(2) * np.random.randn(num_samples))
        return sig + noise


lora = PyLoRa()


