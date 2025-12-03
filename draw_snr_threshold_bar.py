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

def plot_snr_threshold_bar():
    """绘制SNR阈值柱状图"""
    sf_values = [7, 8, 9, 10, 11, 12]
    
    # 使用你数据中的原始方法名，按指定顺序排列
    methods = ["LoRaPHY-CPA", "LoRaPHY-FPA", "HFFT", "MFFT", "LoRa Trimmer", "ChirpSmoother"]
    
    # 颜色设置，与之前保持一致
    colors = {
        "ChirpSmoother": "red",     # 红色
        "HFFT": "#1f77b4",          # 蓝色
        "LoRa Trimmer": "#ff7f0e",  # 橙色
        "LoRaPHY-CPA": "#2ca02c",   # 绿色
        "LoRaPHY-FPA": "#d62728",   # 深红色
        "MFFT": "#9467bd"           # 紫色
    }
    
    # 收集所有数据
    all_data = {}
    for sf in sf_values:
        data = load_sf_data(sf)
        if data is None:
            continue
            
        snr_range = data["snr_range"]
        sf_data = {}
        
        for method in methods:
            if method in data:
                success_rates = data[method]
                threshold_snr = find_snr_threshold(snr_range, success_rates, 0.9)
                sf_data[method] = threshold_snr
        
        all_data[sf] = sf_data
    
    # 绘制柱状图，设置长宽比4:3
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # 设置柱子宽度和位置，柱子变窄70%，增加间隔
    bar_width = 0.084  # 原来0.12的70%
    spacing = 0.016     # 柱子之间的间隔，约为原宽度的10%
    n_methods = len(methods)
    
    # 为每个SF绘制一组柱子
    for i, sf in enumerate(sf_values):
        if sf not in all_data:
            continue
            
        x_base = sf
        
        # 计算每个方法的x位置，包含间隔
        total_width = n_methods * bar_width + (n_methods - 1) * spacing
        x_positions = [x_base - total_width/2 + j * (bar_width + spacing) + bar_width/2 for j in range(n_methods)]
        
        for j, method in enumerate(methods):
            if method in all_data[sf]:
                threshold = all_data[sf][method]
                ax.bar(x_positions[j], threshold, bar_width, 
                      color=colors[method], label=method if i == 0 else "")
    
    # 设置图表属性
    ax.set_xlabel('SF', fontsize=20, fontweight='bold')
    ax.set_ylabel('SNR Threshold (dB)', fontsize=20, fontweight='bold')
    # ax.set_title('SNR Threshold vs Spreading Factor (90% Success Rate)', fontsize=16, fontweight='bold')
    
    # 设置网格
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # 设置x轴
    ax.set_xticks(sf_values)
    ax.set_xticklabels(sf_values, fontsize=16, fontweight='bold')
    
    # 设置y轴
    ax.tick_params(axis='y', labelsize=11)
    
    # 设置图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=20)
    
    # 设置y轴范围，从-35到0
    ax.set_ylim(-35, 0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('output/snr_threshold_bar_chart.png', dpi=300, bbox_inches='tight')
    print("柱状图已保存到 output/snr_threshold_bar_chart.png")

if __name__ == "__main__":
    plot_snr_threshold_bar()