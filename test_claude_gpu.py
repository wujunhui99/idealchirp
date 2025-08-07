from PyLoRa_GPU import PyLoRa
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pytest

# Create a global PyLoRa instance with GPU flag
lora = PyLoRa(use_gpu=True)


# ('Ideal Decode v2', ideal_past, lora.our_ideal_decode_decodev2),
class Config():
    def __init__(self, name, ty, func):
        self.name = name
        self.ty = ty
        self.func = func


configs = [
    Config('Ideal Decode v2', "ideal_past", lora.our_ideal_decode_decodev2),
    Config('LoRa Trimmer', "real_past", lora.loratrimmer_decode),
    Config('LoRaPHY', "real_past", lora.loraphy)
]


def load_sig(file_path):
    return lora.read_file(file_path)


@pytest.mark.parametrize(
    "data, sf,snr_min,snr_max,step,epochs",
    [
        ("mock", 8, -40, -2, 3, 20),
        ("mock", 9, -40, -2, 3, 20),
        ("mock", 10, -40, -2, 3, 20)
    ]
)
def test_multiple_snr(data, sf, snr_min, snr_max, step, epochs):
    lora.sf = sf
    # SNR范围从-30到-3，步进为3
    snr_range = np.arange(snr_min, snr_max, step)
    # 存储每个函数在不同SNR下的准确率
    results = {
        'Ideal Decode v2': [],
        'LoRa Trimmer': [],
        'LoRaPHY': [],
    }

    ideal = os.path.join(".", data, str(sf), "ideal_past")
    real = os.path.join(".", data, str(sf), "real_past")

    # 测试函数配置
    test_configs = [
        ('Ideal Decode v2', ideal, lora.our_ideal_decode_decodev2),
        ('LoRa Trimmer', real, lora.loratrimmer_decode),
        ('LoRaPHY', real, lora.loraphy),
    ]

    # 记录开始时间
    start_time = time.time()

    # 对每个SNR值进行测试
    for snr in snr_range:
        print(f"Testing SNR: {snr}")
        # 测试每个函数
        for name, dir_path, func in test_configs:
            result = 0
            for i in range(2 ** sf):
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

    # 记录结束时间并计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    draw(results, snr_range, sf)
    return results, snr_range, results


def draw(results, snr_range, sf):
    plt.figure(figsize=(10, 6))
    for name in results:
        plt.plot(snr_range, results[name], marker='o', label=name)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title(f'Decoder Performance vs SNR(SF={sf}) mock data')
    plt.grid(True)
    plt.legend()
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    plt.savefig(f'decoder_performance_SF{sf}_' + timestamp + '.png')
    plt.close()