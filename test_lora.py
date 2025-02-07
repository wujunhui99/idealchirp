# 测试不同解码方法性能
import pytest
from pyLoRa import lora
import os

@pytest.mark.parametrize("frequency,expected", [
    (433.0, 433.0),
    (868.0, 868.0),
    (915.0, 915.0)
])
def test_frequency_setting(frequency, expected):
    print(frequency, expected)

def load_sig(file_path):
    return lora.read_file(file_path)
@pytest.mark.parametrize("snr,epochs,file_path,truth,func", [
    (-20, 1000,"./ideal/63.cfile",63,lora.our_ideal_decode_decodev2),
    (-20, 1000,"./real/63.cfile",63,lora.loratrimmer_decode),

])
def test_snr_epoch_trimer(snr,epochs,file_path,truth, func):
    print(snr, epochs, file_path, truth, func)
    sig = load_sig(file_path)
    result = 0
    for i in range(epochs):
        chirp = lora.add_noise(sig=sig, snr=snr)
        ret = func(sig=chirp)[0]
        if ret == truth:
            result += 1
    print(result/epochs)
@pytest.mark.parametrize("snr,epochs,dir_path,func", [
    (-20, 100,"./ideal",lora.our_ideal_decode_decodev2),
    (-20, 100,"./real",lora.loratrimmer_decode),
    (-20, 100,"./real",lora.loraphy_FPA),
    (-20, 100,"./real",lora.loraphy_CPA),

])
def test_snr_epoch_trimer_all(snr,epochs,dir_path, func):
    result = 0
    for i in range(128):
        file_path = os.path.join(dir_path,str(i) + ".cfile")
        truth = i
        sig = load_sig(file_path)

        for i in range(epochs):
            chirp = lora.add_noise(sig=sig, snr=snr)
            ret = func(sig=chirp)[0]
            if ret == truth:
                result += 1
    print(result/(epochs*128))
    return result/(epochs*128)

@pytest.mark.parametrize("snr,epochs,symbol,dir_path,func", [
    (-20, 100,1,"./ideal",lora.our_ideal_decode_decodev2),
    (-20, 100,1,"./real",lora.loratrimmer_decode),
    (-20, 100,1,"./real",lora.loraphy_FPA),
    (-20, 100,1,"./real",lora.loraphy_CPA),

])
def test_snr_epoch_trimer_symbol(snr,epochs,symbol,dir_path, func):
    result = 0

    file_path = os.path.join(dir_path,str(symbol) + ".cfile")
    truth = symbol
    sig = load_sig(file_path)

    for i in range(epochs):
        chirp = lora.add_noise(sig=sig, snr=snr)
        ret = func(sig=chirp)[0]
        if ret == truth:
            result += 1
    print(result/(epochs*128))
    success_rate = result / (epochs * 128)
    assert success_rate >= 0
   #return result/(epochs*128)