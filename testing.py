from pyLoRa import pyLoRa

lora = pyLoRa()

sig = lora.read_file("./mock/ideal/64.cfile")
v2_raw_cnt = 0
v2_bit_cnt = 0
print(lora.our_ideal_decode_decodev2_bit(sig=sig))
print(lora.our_ideal_decode_decodev2(sig=sig))


for i in range(1000):
    sign = lora.add_noise(sig = sig,snr=-25)
    v2_bit = lora.our_ideal_decode_decodev2_bit(sig=sign)[0]
    v2_raw = lora.our_ideal_decode_decodev2(sig=sign)[0]
    if(v2_bit == 16):
        v2_bit_cnt += 1
    if(v2_raw == 64):
        v2_raw_cnt += 1
print(v2_bit_cnt)
print(v2_raw_cnt)

