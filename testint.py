from PyLoRa import PyLoRa
import numpy as np
lora = PyLoRa()
lora.sf = 7
up_chirp = lora.ideal_chirpx(f0=21,rate=1)
zeros = np.zeros(0 * len(up_chirp), dtype=complex)
down_chirp = lora.ideal_chirp(f0=0,iq_invert=1)
zp_u = np.concatenate((up_chirp, zeros))
zp_d = np.concatenate((down_chirp, zeros))
dechirped = zp_u * zp_d
freq = np.fft.fft(dechirped)
amp =  np.abs(freq)
print(amp)

up_chirp = lora.ideal_chirpx(f0=17,rate=2)
zeros = np.zeros(1 * len(up_chirp), dtype=complex)
down_chirp = lora.ideal_chirp(f0=0,iq_invert=1)
zp_u = np.concatenate((up_chirp, zeros))
zp_d = np.concatenate((down_chirp, zeros))
dechirped = zp_u * zp_d
freq = np.fft.fft(dechirped)
amp =  np.abs(freq)
print(amp)