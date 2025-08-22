import json
import matplotlib.pyplot as plt
import numpy as np
import os

def find_snr_threshold(snr_values, success_rates, threshold=0.9):
    """找到达到指定成功率阈值的SNR值"""
    for i, rate in enumerate(success_rates):
        if rate >= threshold:
            if i == 0:
                return snr_values[0]
            else:
                # 线性插值
                prev_snr = snr_values[i-1]
                curr_snr = snr_values[i]
                prev_rate = success_rates[i-1]
                curr_rate = success_rates[i]
                
                # 线性插值计算精确的SNR阈值
                snr_threshold = prev_snr + (threshold - prev_rate) * (curr_snr - prev_snr) / (curr_rate - prev_rate)
                return snr_threshold
    
    # 如果没有达到阈值，返回最后一个SNR值
    return snr_values[-1]

def load_sf_data(sf):
    """加载指定SF的数据"""
    file_path = f"output/SF{sf}/sf{sf}.json"
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_snr_threshold_vs_sf():
    """绘制SNR阈值与SF的关系图"""
    sf_values = [7, 8, 9, 10, 11, 12]
    methods = ["ChirpSmoother", "LoRa Trimmer", "LoRaPHY-CPA", "LoRaPHY-FPA", "MFFT", "HFFT"]
    
    # 存储每个方法在不同SF下的SNR阈值
    method_thresholds = {method: [] for method in methods}
    valid_sf_values = []
    
    for sf in sf_values:
        data = load_sf_data(sf)
        if data is None:
            print(f"Warning: No data found for SF{sf}")
            continue
            
        valid_sf_values.append(sf)
        snr_range = data["snr_range"]
        
        for method in methods:
            if method in data:
                success_rates = data[method]
                threshold_snr = find_snr_threshold(snr_range, success_rates, 0.9)
                method_thresholds[method].append(threshold_snr)
            else:
                method_thresholds[method].append(None)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, method in enumerate(methods):
        # 过滤掉None值
        sf_filtered = []
        threshold_filtered = []
        for j, threshold in enumerate(method_thresholds[method]):
            if threshold is not None:
                sf_filtered.append(valid_sf_values[j])
                threshold_filtered.append(threshold)
        
        if sf_filtered:
            plt.plot(sf_filtered, threshold_filtered, 
                    marker=markers[i], color=colors[i], 
                    linewidth=2, markersize=8, label=method)
    
    plt.xlabel('Spreading Factor (SF)', fontsize=12)
    plt.ylabel('SNR Threshold (dB)', fontsize=12)
    plt.title('SNR Threshold vs Spreading Factor (90% Decoding Success Rate)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(valid_sf_values)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('output/snr_threshold_vs_sf.png', dpi=300, bbox_inches='tight')
    print("图片已保存到 output/snr_threshold_vs_sf.png")

if __name__ == "__main__":
    plot_snr_threshold_vs_sf()