from pyLoRa import pyLoRa
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
lora = pyLoRa()
def load_sig(file_path):
    return lora.read_file(file_path)


def test_multiple_snr():
    # SNR范围从-30到-3，步进为3
    snr_range = np.arange(-30, -2, 3)
    epochs = 10  # 可以根据需要调整

    # 存储每个函数在不同SNR下的准确率
    results = {
        'Ideal Decode v2': [],
        'Ideal Decode v2 5bit': [],
        # 'Ideal Decode v3': [],
        # 'Ideal Decode': [],
        'LoRa Trimmer': [],
        # 'LoRa filterTrimmer': [],
        # 'LoRa filter2Trimmer': [],
        'LoRaPHY': [],
        # 'Ideal Decode v2x': [],
        # 'Ideal Decode v2z': []

    }
    
    # 测试函数配置
    test_configs = [
        ('Ideal Decode v2', './mock/ideal', lora.our_ideal_decode_decodev2),
        ('Ideal Decode v2 5bit', './mock/ideal', lora.our_ideal_decode_decodev2_bit),
        # ('Ideal Decode v3', './mock/ideal', lora.our_ideal_decode_decodev2),
        # ('Ideal Decode', './mock/ideal', lora.our_ideal_decode),
        ('LoRa Trimmer', './mock/real', lora.loratrimmer_decode),
        # ('LoRa filterTrimmer', './mock/real', lora.filter_loratrimmer_decode),
        # ('LoRa filter2Trimmer', './mock/real', lora.filter2_loratrimmer_decode),
        ('LoRaPHY', './mock/real', lora.loraphy),
        # ('Ideal Decode v2x', './mock/ideal', lora.our_ideal_decode_decodev2x),
        # ('Ideal Decode v2z', './mock/ideal', lora.our_ideal_decode_decodev2z),
    ]

    # 对每个SNR值进行测试
    for snr in snr_range:
        print(f"Testing SNR: {snr}")

        # 测试每个函数
        for name, dir_path, func in test_configs:
            if name == "Ideal Decode v2 5bit":
                result = 0
                for i in range(2 ** lora.bit):
                    idx = i *(2 ** (lora.sf - lora.bit))
                    file_path = os.path.join(dir_path, str(idx) + ".cfile")
                    truth = i
                    sig = load_sig(file_path)
                    for _ in range(epochs):
                        chirp = lora.add_noise(sig=sig, snr=snr)
                        ret = func(sig=chirp)[0]
                        if ret == truth:
                            result += 1
                accuracy = result / (epochs * (2 ** lora.bit))
                results[name].append(accuracy)
                print(f"{name}: {accuracy:.4f}")

            else:
                result = 0
                for i in range(128):
                    file_path = os.path.join(dir_path, str(i) + ".cfile")
                    truth = i
                    sig = load_sig(file_path)

                    for _ in range(epochs):
                        chirp = lora.add_noise(sig=sig, snr=snr)
                        ret = func(sig=chirp)[0]
                        if ret == truth:
                            result += 1

                accuracy = result / (epochs * 128)
                results[name].append(accuracy)
                print(f"{name}: {accuracy:.4f}")

    # 绘制结果
    plt.figure(figsize=(10, 6))
    for name in results:
        plt.plot(snr_range, results[name], marker='o', label=name)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Decoder Performance vs SNR(SF=7) mock data')
    plt.grid(True)
    plt.legend()
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    plt.savefig('decoder_performance' + timestamp + '.png')
    plt.close()

    return results


if __name__ == "__main__":
    t0 = time.time()
    results = test_multiple_snr()
    print(results)
    t1 = time.time()
    print(f"{t1 - t0:.4f} seconds")