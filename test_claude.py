import pytest
from pyLoRa import lora
import os
import numpy as np
import matplotlib.pyplot as plt


def load_sig(file_path):
    return lora.read_file(file_path)


def test_multiple_snr():
    # SNR范围从-30到-3，步进为3
    snr_range = np.arange(-30, -2, 3)
    epochs = 100  # 可以根据需要调整

    # 存储每个函数在不同SNR下的准确率
    results = {
        'Ideal Decode v2': [],
        'Ideal Decode': [],
        'LoRa Trimmer': [],
        'LoRa FPA': [],
        'LoRa CPA': []
    }

    # 测试函数配置
    test_configs = [
        ('Ideal Decode v2', './ideal', lora.our_ideal_decode_decodev2),
        ('Ideal Decode', './ideal', lora.our_ideal_decode),
        ('LoRa Trimmer', './real', lora.loratrimmer_decode),
        ('LoRa FPA', './real', lora.loraphy_FPA),
        ('LoRa CPA', './real', lora.loraphy_CPA)
    ]

    # 对每个SNR值进行测试
    for snr in snr_range:
        print(f"Testing SNR: {snr}")

        # 测试每个函数
        for name, dir_path, func in test_configs:
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
    plt.title('Decoder Performance vs SNR')
    plt.grid(True)
    plt.legend()
    plt.savefig('decoder_performance.png')
    plt.close()

    return results


if __name__ == "__main__":
    results = test_multiple_snr()