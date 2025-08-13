from PyLoRa import  PyLoRa

lora = PyLoRa()


x = 0
sig_ideal = lora.ideal_chirp(f0 = x)
sig_real = lora.ideal_chirp(f0 = x)
trimmer = 0
trimmer_edit = 0
smoother = 0

hfft = 0
for i in range(400):
    sig_real_nosia = lora.add_noise(sig_real,snr=-20)
    sig_ideal_nosia = lora.add_noise(sig_ideal,snr=-20)
    sig_hfft = lora.add_noise(sig_ideal,snr=-20)
    ret_ideal = lora.fft_ideal_decode_decodev2(sig_ideal_nosia)
    ret_real = lora.loratrimmer_decode(sig_ideal_nosia)
    ret_hfft = lora.hfft_decode(sig_hfft)
    ret_trimmer_edit = lora.lora_trimmer_edit(sig_real_nosia)
    if(ret_ideal[0] == x):
        smoother += 1
    if(ret_real[0] == x):
        trimmer += 1
    if(ret_hfft[0] == x):
        hfft += 1
    if(ret_trimmer_edit[0] == x):
        trimmer_edit += 1

print(trimmer)
print(trimmer_edit)
print(smoother)
print(hfft)




