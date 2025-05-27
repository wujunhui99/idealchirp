from PyLoRa import  PyLoRa

lora = PyLoRa()

sig = lora.ideal_chirp(f0 = 67)
lora.our_idealx_decode_decodev2(sig)


