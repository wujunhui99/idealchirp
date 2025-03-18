import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
def draw(results,snr_range,sf):
    plt.figure(figsize=(10, 6))
    for name in results:
        plt.plot(snr_range, results[name], marker='o', label=name)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title(f'Decoder Performance vs SNR(SF={sf}) mock data')
    plt.grid(True)
    plt.legend()
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    plt.savefig(f'./imgs/decoder_performance_SF{sf}_' + timestamp + '.png')
    plt.close()
import json

# 打开 JSON 文件并读取内容

with open('past/nSF10.json', 'r', encoding='utf-8') as file:
    data = json.load(file)  # 反序列化 JSON 文件内容为 Python 对象
print("JSON 文件内容已成功加载！")

snr_range = np.arange(-40, -2, 1)
sf = data['sf']
del data['sf']
del data['snr_range']
del data['mfft_decode']

draw(data,snr_range,sf)