from PyLoRa import lora
from scipy import fft
import numpy as np
sig = lora.real_chirp(f0=60)

sign = lora.add_noise(sig,snr = -8)

downchirp = lora.real_chirp(f0=0,iq_invert=1)

dechirped = sign * downchirp

res = fft.fft(dechirped)

amp = np.abs(res)
amp = amp[:128]

# print(amp)

import matplotlib.pyplot as plt
import numpy as np

# 假设你的amp数组已经存在
# amp = amp[:128]

# 方法1：最简单的线图
plt.figure(figsize=(10, 6))
plt.plot(amp)
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Amplitude vs Index')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

# plt.subplot(amp)
# plt.xlabel('Index')
# plt.ylabel('Amplitude')
# plt.title('Amplitude vs Index')
# plt.grid(True)


# 三次样条插值
# plt.subplot(2, 2, 2)
# x_original = np.arange(len(amp))
# x_smooth = np.linspace(0, len(amp)-1, len(amp)*4)  # 增加4倍数据点
# f = interpolate.interp1d(x_original, amp, kind='cubic')
# amp_smooth = f(x_smooth)
# plt.plot(x_smooth, amp_smooth)
# plt.title('三次样条插值')
# plt.grid(True)
#
# plt.figure(figsize=(10, 6))
# amp_gaussian = gaussian_filter1d(amp, sigma=1.5)
# plt.plot(amp_gaussian)
# plt.title('高斯滤波平滑')
# plt.grid(True)
#
# plt.figure(figsize=(10, 6))
# amp_smooth_combined = gaussian_filter1d(amp, sigma=1.0)
# x_original = np.arange(len(amp_smooth_combined))
# x_interp = np.linspace(0, len(amp_smooth_combined)-1, len(amp_smooth_combined)*3)
# f = interpolate.interp1d(x_original, amp_smooth_combined, kind='cubic')
# amp_final = f(x_interp)
# plt.plot(x_interp, amp_final, linewidth=2)
# plt.xlabel('Index')
# plt.ylabel('Amplitude')
# plt.title('平滑后的振幅图')
# plt.grid(True, alpha=0.3)
# plt.show()
# plt.show()
# plt.show()

# 高斯滤波 + 插值的组合，然后旋转90度
# plt.figure(figsize=(6, 10))  # 调整图形尺寸适应旋转后的形状
# amp_smooth_combined = gaussian_filter1d(amp, sigma=1.5)
# x_original = np.arange(len(amp_smooth_combined))
# x_interp = np.linspace(0, len(amp_smooth_combined)-1, len(amp_smooth_combined)*3)
# f = interpolate.interp1d(x_original, amp_smooth_combined, kind='cubic')
# amp_final = f(x_interp)
#
# # 交换x轴和y轴
# plt.plot(amp_final, x_interp, linewidth=2)
# plt.ylabel('Index')
# plt.xlabel('Amplitude')
# plt.title('旋转后的振幅图')
# plt.show()

#
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import interpolate
# from scipy.ndimage import gaussian_filter1d
#
# # 假设你的amp数组已经存在
# # amp = amp[:128]
#
# # 高斯滤波 + 插值的组合，然后旋转90度
# plt.figure(figsize=(6, 10))
# amp_smooth_combined = gaussian_filter1d(amp, sigma=1.5)
# x_original = np.arange(len(amp_smooth_combined))
# x_interp = np.linspace(0, len(amp_smooth_combined)-1, len(amp_smooth_combined)*3)
# f = interpolate.interp1d(x_original, amp_smooth_combined, kind='cubic')
# amp_final = f(x_interp)
#
# # 交换x轴和y轴
# plt.plot(amp_final, x_interp, linewidth=2)
#
# # 自定义y轴标注
# max_y = x_interp[-1]  # 获取y轴的最大值
# y_ticks = [0, max_y/4, max_y/2, max_y]  # 四个刻度位置
# y_labels = ['0', 'B/4', 'B/2','3B/4', 'B']     # 四个刻度标签
#
# plt.yticks(y_ticks, y_labels)
# plt.ylabel('Frequency')
# # frequency
# plt.xlabel('Amplitude')
# plt.title('旋转后的振幅图')
# plt.show()
#
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import interpolate
# from scipy.ndimage import gaussian_filter1d
#
# # 设置中文字体，解决中文显示问题
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 假设你的amp数组已经存在
# # amp = amp[:128]
#
# # 高斯滤波 + 插值的组合，然后旋转90度
# plt.figure(figsize=(6, 10))
# amp_smooth_combined = gaussian_filter1d(amp, sigma=1.0)
# x_original = np.arange(len(amp_smooth_combined))
# x_interp = np.linspace(0, len(amp_smooth_combined)-1, len(amp_smooth_combined)*3)
# f = interpolate.interp1d(x_original, amp_smooth_combined, kind='cubic')
# amp_final = f(x_interp)
#
# # 交换x轴和y轴
# plt.plot(amp_final, x_interp, linewidth=2)
#
# # 自定义y轴标注 - 5个点
# max_y = x_interp[-1]  # 获取y轴的最大值
# y_ticks = [0, max_y/4, max_y/2, 3*max_y/4, max_y]  # 五个刻度位置
# y_labels = ['0', 'B/4', 'B/2', '3B/4', 'B']        # 五个刻度标签
#
# plt.yticks(y_ticks, y_labels)
# plt.ylabel('Frequency')
# plt.xlabel('Amplitude')
# plt.title('旋转后的振幅图')
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

# 假设你的amp数组已经存在
# amp = amp[:128]

# 高斯滤波 + 插值的组合，然后旋转90度
plt.figure(figsize=(6, 10))
amp_smooth_combined = gaussian_filter1d(amp, sigma=1.5)
x_original = np.arange(len(amp_smooth_combined))
x_interp = np.linspace(0, len(amp_smooth_combined)-1, len(amp_smooth_combined)*3)
f = interpolate.interp1d(x_original, amp_smooth_combined, kind='cubic')
amp_final = f(x_interp)

# 交换x轴和y轴
plt.plot(amp_final, x_interp, linewidth=2)

# 自定义y轴标注 - 5个点，显示文字标签
max_y = x_interp[-1]   # 获取y轴的最大值
y_ticks = [0, max_y/4, max_y/2, 3*max_y/4, max_y]  # 五个刻度位置
y_labels = ['0', 'B/4', 'B/2', '3B/4', 'B']        # 五个刻度标签

# plt.yticks(y_ticks, ('0', 'B/4', 'B/2', '3B/4', 'B'))
plt.yticks((0,31,63,95,127),('0', 'B/4', 'B/2', '3B/4', 'B'))
plt.ylabel('Frequency')
plt.xlabel('Amplitude')
plt.title('Rotated Amplitude Plot')
plt.show()