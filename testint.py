# 测试f_0 = 0情况下的解码性能
from PyLoRa import  PyLoRa

lora = PyLoRa()


x = 0
sig_ideal = lora.ideal_chirp(f0 = x)
sig_real = lora.ideal_chirp(f0 = x)
trimmer = 0
trimmer_edit = 0
smoother = 0
smoother_trimmer = 0
hfft = 0
for i in range(1000):
    sig_real_nosia = lora.add_noise(sig_real,snr=-20)
    sig_ideal_nosia = lora.add_noise(sig_ideal,snr=-20)
    sig_hfft = lora.add_noise(sig_ideal,snr=-20)
    ret_ideal = lora.fft_ideal_decode_decodev2(sig_ideal_nosia)
    ret_real = lora.loratrimmer_decode(sig_ideal_nosia)
    ret_hfft = lora.hfft_decode(sig_hfft)
    ret_smoother_trimmer = lora.chirpsmoother_decodex(sig_ideal_nosia)

    if(ret_ideal[0] == x):
        smoother += 1
    if(ret_real[0] == x):
        trimmer += 1
      
    if(ret_smoother_trimmer[0] == x):
        smoother_trimmer += 1

print(trimmer)
print(smoother)
print(hfft)
print(smoother_trimmer)




