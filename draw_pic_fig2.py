from PyLoRa import lora
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

sig = lora.real_chirp(f0=48)
sigl = lora.ideal_chirp(f0=48)

sign = lora.add_noise(sig,snr = -4)
sigln = sign -sig + sigl

lora.write_file(file_path="nosia.cfile",sig = sigln)

downchirp = lora.real_chirp(f0=0,iq_invert=1)

dechirped = sign * downchirp

res = fft.fft(dechirped)

amp0 = np.abs(res)
amppos = amp0[:128]
ampneg = amp0[len(amp0)-128:]

#amp = amppos + ampneg

amp = np.hstack((amppos,ampneg))


def plot_pic(size,x_ticks ,x_lables,y_ticks,amp,file_name, title = "testing",swap_axes=False):
    if(swap_axes):
        size = (size[1], size[0])
    plt.figure(figsize=size)
    amp_smooth_combined = gaussian_filter1d(amp, sigma=1.8)
    x_original = np.arange(len(amp_smooth_combined))
    x_interp = np.linspace(0, len(amp_smooth_combined) - 1, len(amp_smooth_combined) * 3)
    f = interpolate.interp1d(x_original, amp_smooth_combined, kind='cubic')
    amp_final = f(x_interp)

    if swap_axes:
        plt.plot(amp_final, x_interp, linewidth=2, color='black')
    else:
        plt.plot(x_interp, amp_final, linewidth=2, color='black')

    plt.xticks([])
    plt.yticks([])
    plt.tick_params(axis='both', which='both', length=0)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(file_name + '.png')
    plt.show()

size = (8,8)
x_ticks =np.linspace(0,len(amp) - 1,5)
x_labels = ['0', 'B/4', 'B/2', '3B/4', 'B']
y_ticks = (0, 100, 200, 300, 400, 500, 600)
file_name = "res2"
title = "(b)"
# size = (16,8)
# amp = np.concatenate((ampneg,amppos))
# x_ticks =np.linspace(0,len(amp) - 1,9)
# x_labels = ['-B','-B/4','-B/2','-3B/4','0', 'B/4', 'B/2', '3B/4', 'B']
# y_ticks = (0, 100, 200, 300)
# file_name = "res"
# title = "(a)"
plot_pic(size,x_ticks,x_labels,y_ticks,amp,file_name,title=title,swap_axes= 0 )