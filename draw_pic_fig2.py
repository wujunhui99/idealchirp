from PyLoRa import lora
import numpy as np

sig1 = lora.real_chirp(f0=63)
sign1 = lora.add_noise(sig=sig1,snr = 3)

sig2 = lora.real_chirp(f0=63)
sign2 = lora.add_noise(sig=sig2,snr = 3)
sign = np.concatenate((sign1,sign2))
lora.write_file(sig=sign,file_path="fig2.cfile")

sigd1 = sign1 * lora.real_chirp(f0=0,iq_invert=1)
sigd2 = sign1 * lora.real_chirp(f0=0,iq_invert=1)
sigd = np.concatenate((sigd1,sigd2))
lora.write_file(sig=sigd,file_path="dec.cfile")