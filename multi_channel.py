import math

from PyLoRa import PyLoRa
import numpy as np
import Tools
lora = PyLoRa()

def sig_for_test():
    sig0 = lora.ideal_chirp(f0=0)
    sig1 = lora.ideal_chirp(f0=0)
    sig0 = Tools.freq_shift(sig0, -400e3,1e6)
    sig1 = Tools.freq_shift(sig1, -200e3,1e6)
    zeros = Tools.zeros(len(sig0))
    sig0 = Tools.merge(sig0,zeros)
    sig1 = Tools.merge(sig1,zeros)
    sig1 = Tools.right_move(sig=sig1,n = len(sig1) / 2 * 80 / 125)
    sigx = 1 * sig0 + 1 * 4 * sig1
    lora.write_file("merge.cfile",sig=sigx)
    return sigx

def upcacalate_part(maxBin):
    return ((75/125 * 128) + 128 - maxBin) * 8
def downcacalate_part(maxBin):
    return 1024 - ((75/125 * 128) + 128 - maxBin) * 8
sigx = sig_for_test()
sigx = Tools.freq_shift(sigx, 400e3,1e6)
lora.write_file(sig=sigx,file_path="conflict.cfile")

def multi_channle(sigx):
    # Work on a copy to avoid in-place mutations affecting subsequent calls
    sig = Tools.trunck_sig(sigx, int(lora.fs * 2 ** lora.sf / lora.bw)).copy()
    idx0, power = lora.our_ideal_decode_decodev2(sig=sig)
    idxr_current = idx0
    part = int(upcacalate_part(idx0))
    if(part < 1024):
        idxr, pow = lora.subr_our_ideal_decode_decodev2(sig=sig, n=int(part))
        idxl, powl =  lora.subl_our_ideal_decode_decodev2(sig=sig, n=int(part))
        if(idxr != idxl):
            fft_bin = lora.subr_raw_our_ideal_decode_decodev2(sig=sig, n=int(part))[idxr]
            pw = fft_bin / (lora.get_samples_per_symbol() - int(part))
            rev = -lora.ideal_chirp(f0=idxr) * pw
            rev[:part] = 0
            # Avoid in-place modify of potential views
            sig = sig + rev
            idxr_current = idxr
    part = int(downcacalate_part(idxr_current))
    if (part >= 0):
        idxl, pow = lora.subl_our_ideal_decode_decodev2(sig=sig, n=int(part))
        idxr, powr = lora.subr_our_ideal_decode_decodev2(sig=sig, n=int(part))
        if (idxr != idxl):
            fft_bin = lora.subl_raw_our_ideal_decode_decodev2(sig=sig, n=int(part))[idxl]
            pw = fft_bin / (int(part))
            rev = -lora.ideal_chirp(f0=idxl) * pw
            rev[:part] = 0
            sig += rev
    return lora.our_ideal_decode_decodev2(sig=sig)
    pass

import time

t0 = time.time()
for i in range(100):
    multi_channle(sigx)
t1 = time.time()
print(t1- t0)

t0 = time.time()
for i in range(100):
    lora.our_ideal_decode_decodev2(sigx)

print(time.time() - t0)






