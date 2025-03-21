from PyLoRa import PyLoRa
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pytest
import json
from datetime import datetime

class Singleton:
    _instance = None
    _folder_name = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y%m%d%H%M%S")
            folder_name = f"./output/record{formatted_time}"
            os.makedirs(folder_name)
            cls._folder_name = folder_name
        return cls._instance
    def get_folder(self):
        return self._folder_name


lora = PyLoRa()
#('Ideal Decode v2', ideal, lora.our_ideal_decode_decodev2),

def load_sig(file_path):
    return lora.read_file(file_path)

@pytest.mark.parametrize(
    "data, sf,snr_min,snr_max,step,epochs",
    [
        ("mock", 7, -40, -2, 2, 8),
        ("mock",8,-40,-2,2,4),
        ("mock",9,-40,-2,2,2),
        ("mock",10,-40,-2,2,1)
    ]
)
def test_multiple_snr(data, sf,snr_min,snr_max,step,epochs):
    lora.sf = sf
    # SNR范围从-30到-3，步进为3
    snr_range = np.arange(snr_min, snr_max, step)
    # epochs = 1  # 可以根据需要调整
    # 存储每个函数在不同SNR下的准确率
    results = {
        'Ideal Decode v2': [],
        'hfft_decode':[],
        # 'mfft_decode':[],
        # 'Ideal Decode v2 5bit': [],
        'LoRa Trimmer': [],
        'LoRaPHY': [],
    }

    ideal = os.path.join("./datasets",data,str(sf),"ideal")
    real = os.path.join("./datasets",data,str(sf),"real")
    # 测试函数配置
    test_configs = [
        ('Ideal Decode v2', ideal, lora.our_ideal_decode_decodev2),
        ('hfft_decode', real, lora.hfft_decode),
        # ('mfft_decode', real, lora.mfft_decode),
        ('LoRa Trimmer', real, lora.loratrimmer_decode),
        ('LoRaPHY', real, lora.loraphy),
    ]

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

            accuracy = result / (epochs * 2 ** sf)
            results[name].append(accuracy)
            print(f"{name}: {accuracy:.4f}")
    s1 = Singleton()
    # 创建文件夹路径
    folder_name = s1.get_folder()
    draw(results,snr_range,sf,folder_name)
    with open(os.path.join(folder_name,f'./sf{sf}.json'), 'w') as file:
        res = results.copy()
        res['snr_range'] = snr_range.tolist()
        res['sf'] = sf
        json.dump(res, file, indent=4)
    return results, snr_range, results
def draw(results,snr_range,sf,folder_name):
    plt.figure(figsize=(10, 6))
    for name in results:
        plt.plot(snr_range, results[name], marker='o', label=name)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title(f'Decoder Performance vs SNR(SF={sf}) mock data')
    plt.grid(True)
    plt.legend()
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    s1 = Singleton()
    folder_name = s1.get_folder()
    plt.savefig(os.path.join(folder_name,f'decoder_performance_SF{sf}'  + '.png'))
    plt.close()

