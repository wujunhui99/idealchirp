from PyLoRa import  PyLoRa

lora = PyLoRa()

sig = lora.ideal_chirp(f0=0)

# 1
lora.our_ideal_decode_decodev2(sig)
# 2
lora.fft_ideal_decode_decodev2(sig)

sig2 = lora.real_chirp(f0=0)

# 3
lora.MFFT(sig)
# 4
lora.hfft_decode(sig2)
# 5
lora.loraphy_fpa(sig2)
# 6
lora.loraphy_cpa(sig2)
#7
lora.loratrimmer_decode(sig2)
