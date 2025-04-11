import numpy as np
from PyLoRa import PyLoRa
# 假设你的原始信号数组为signal
# 采样率为fs (Hz)
lora = PyLoRa()
signal = lora.ideal_chirp(f0=0)
fs = 1000000  # 示例采样率1MHz
dt = 1/fs  # 采样间隔
t = np.arange(len(signal)) * dt  # 时间数组

# 要提升的频率(Hz)
f_shift = 200000  # 200kHz

# 生成频率搬移因子
shift_factor = np.exp(1j * 2 * np.pi * f_shift * t)

# 进行频率搬移
shifted_signal = signal * shift_factor

sig2 = lora.ideal_chirp(f0=0)
add_sig = sig2 + shifted_signal
lora.write_file(sig=add_sig,file_path="d.cfile")