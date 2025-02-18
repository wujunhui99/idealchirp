from pyLoRa import pyLoRa

lora = pyLoRa()

lora.sf = 8
sig = lora.ideal_chirp(f0=128)
