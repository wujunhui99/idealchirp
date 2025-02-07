# 测试不同symbol的解码性能
from pyLoRa import lora
import os
import numpy as np
import matplotlib.pyplot as plt


def load_sig(file_path):
    return lora.read_file(file_path)


def test_symbol_performance():
    snr = -20
    epochs = 400
    symbols = range(128)

    # 更新结果存储字典
    results = {
        'Ideal Decode v2': [],
        'Ideal Decode': [],
        'LoRa Trimmer': [],
        'LoRa FPA': []
    }

    # 更新测试配置
    test_configs = [
        ('Ideal Decode v2', './ideal', lora.our_ideal_decode_decodev2),
        ('Ideal Decode', './ideal', lora.our_ideal_decode),
        ('LoRa Trimmer', './real', lora.loratrimmer_decode),
        ('LoRa FPA', './real', lora.loraphy_FPA)
    ]

    # 对每个symbol进行测试
    for symbol in symbols:
        if symbol % 10 == 0:
            print(f"Testing symbol: {symbol}")

        for name, dir_path, func in test_configs:
            result = 0
            file_path = os.path.join(dir_path, str(symbol) + ".cfile")
            truth = symbol
            sig = load_sig(file_path)

            for _ in range(epochs):
                chirp = lora.add_noise(sig=sig, snr=snr)
                ret = func(sig=chirp)[0]
                if ret == truth:
                    result += 1

            accuracy = result / epochs
            results[name].append(accuracy)

    # 绘制结果
    plt.figure(figsize=(15, 8))

    # 定义不同线型和颜色
    styles = [
        ('Ideal Decode v2', 'b-', 'blue'),
        ('Ideal Decode', 'r--', 'red'),
        ('LoRa Trimmer', 'g:', 'green'),
        ('LoRa FPA', 'm-.', 'magenta')
    ]

    # 使用不同的线型绘制每个方法的结果
    for name, style, color in styles:
        plt.plot(symbols, results[name], style, label=name, alpha=0.8, linewidth=2)

    plt.xlabel('Symbol', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Decoder Performance vs Symbol (SNR = {snr}dB)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 设置x轴刻度
    plt.xticks(np.arange(0, 128, 16))

    # 设置y轴范围从0到1
    plt.ylim(0, 1.1)

    # 添加网格
    plt.grid(True, which='both', linestyle='--', alpha=0.2)

    plt.savefig('decoder_symbol_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results


if __name__ == "__main__":
    results = test_symbol_performance()