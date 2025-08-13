# 测试不同symbol的解码性能
from PyLoRa import PyLoRa
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.ndimage import gaussian_filter1d
lora = PyLoRa()

def load_sig(file_path):
    return lora.read_file(file_path)


def test_symbol_performance():
    snr = -20
    epochs = 500
    symbols = range(128)

    # 更新结果存储字典
    results = {
        'ChirpSmoother': [],
        'LoRa Trimmer': [],
        'LoRaPHY-CPA': [],
        'LoRaPHY-FPA':[],
        'HFFT':[],
        'MFFT':[]
    }

    # 更新测试配置

    test_configs = [
        ('ChirpSmoother', './datasets/mock/7/our', lora.fft_ideal_decode_decodev2),
        ('LoRa Trimmer', './datasets/mock/7/tradition', lora.loratrimmer_decode),
        ('LoRaPHY-CPA', './datasets/mock/7/tradition', lora.loraphy),
        ('LoRaPHY-FPA', './datasets/mock/7/tradition', lora.loraphy_fpa),
        ('HFFT', './datasets/mock/7/tradition', lora.hfft_decode),
        ('MFFT', './datasets/mock/7/tradition', lora.MFFT)
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

    # 保存结果到JSON文件
    with open('symbols.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to symbols.json")

    # 绘制结果
    plt.figure(figsize=(16, 12))



    # 定义不同线型和颜色（学术论文风格）
    styles = [
        ('ChirpSmoother', 'red', 'solid'),

        ('HFFT', '#1f77b4', 'solid'),  # 蓝色
        ('LoRa Trimmer', '#ff7f0e', 'dashed'),  # 橙色
        ('LoRaPHY-CPA', '#2ca02c', 'dotted'),  # 绿色
        ('LoRaPHY-FPA', '#d62728', 'dashdot'),  # 深红色
        ('MFFT', '#9467bd', 'solid')  # 紫色
    ]

    # 使用不同的线型绘制每个方法的结果
    for name, color, linestyle in styles:
        if name in results:
            plt.plot(symbols, results[name], color=color, linestyle=linestyle, label=name, alpha=0.8, linewidth=2)

    plt.xlabel('Symbol', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 设置x轴刻度为符号表示
    B = lora.bw  # 带宽
    tick_positions = [0, 32, 64, 96, 128]  # 对应于-B/2, -B/4, 0, B/4, B/2
    tick_labels = ['-B/2', '-B/4', '0', 'B/4', 'B/2']
    plt.xticks(tick_positions, tick_labels)

    # 设置y轴范围从0到1
    plt.ylim(0, 1.1)

    # 添加网格
    plt.grid(True, which='both', linestyle='--', alpha=0.2)

    plt.savefig('decoder_symbol_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results


def draw_from_json(json_file='symbols.json', snr=-20):
    """
    从JSON文件中读取结果并绘图
    
    Args:
        json_file: JSON文件路径，默认为'symbols.json'
        snr: SNR值，用于文件命名，默认为-20
    """
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Please run test_symbol_performance() first.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file}")
        return
    
    symbols = range(128)
    
    # 绘制结果
    plt.figure(figsize=(16, 12))


    # 定义不同线型和颜色（学术论文风格）
    styles = [
        ('ChirpSmoother', 'red', 'solid'),

        ('HFFT', '#1f77b4', 'solid'),  # 蓝色
        ('LoRa Trimmer', '#ff7f0e', 'dashed'),  # 橙色
        ('LoRaPHY-CPA', '#2ca02c', 'dotted'),  # 绿色
        ('LoRaPHY-FPA', '#d62728', 'dashdot'),  # 深红色
        ('MFFT', 'cyan', 'solid')  # 紫色
    ]

    # 使用不同的线型绘制每个方法的结果
    for name, color, linestyle in styles:

        if name in results:
            plt.plot(symbols, results[name], color=color, linestyle=linestyle, label=name, alpha=0.8, linewidth=2)

    plt.xlabel('Symbol', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 设置x轴刻度为符号表示
    B = lora.bw  # 带宽
    tick_positions = [0, 32, 64, 96, 128]  # 对应于-B/2, -B/4, 0, B/4, B/2
    tick_labels = ['-B/2', '-B/4', '0', 'B/4', 'B/2']
    plt.xticks(tick_positions, tick_labels)

    # 设置y轴范围从0到1
    plt.ylim(0, 1.1)

    # 添加网格
    plt.grid(True, which='both', linestyle='--', alpha=0.2)

    plt.savefig('decoder_symbol_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved from JSON data to decoder_symbol_performance.png")


if __name__ == "__main__":
    # 可以选择运行测试或从JSON绘图
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'draw':
        draw_from_json()
    else:
        results = test_symbol_performance()