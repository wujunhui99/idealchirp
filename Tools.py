import numpy as np

def reverse(sig,beg,end):
    while(end > beg):
        sig[end],sig[beg] = sig[beg], sig[end]
        end -= 1
        beg += 1
    return sig
def right_move(sig, n):
    n = int(n)
    reverse(sig,0,len(sig)-1)
    reverse(sig,0,n-1)
    reverse(sig,n,len(sig)-1)
    return sig

def freq_shift(sig,freq, fs):
    t = np.arange(0,len(sig) / fs, 1 / fs)
    shift_signal = np.exp(1j * 2 * np.pi * freq * t)
    return shift_signal * sig

def zeros(n):
    zeros = np.zeros(n, dtype=complex)
    return zeros

def merge(sig1, sig2):
    return np.hstack((sig1,sig2))

def trunck_sig(sig,size):
    return sig[:size]
