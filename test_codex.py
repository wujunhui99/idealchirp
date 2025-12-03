import sys
# 动态选择 CPU/GPU 版本的 PyLoRa，默认 CPU，命令行包含 'gpu' 时启用 GPU 版本
_use_gpu = any(arg.lower() == 'gpu' for arg in sys.argv[1:])
try:
    if _use_gpu:
        from PyLoRa_GPU import PyLoRa as _PyLoRaSelected
        SELECTED_BACKEND = 'gpu'
    else:
        from PyLoRa import PyLoRa as _PyLoRaSelected
        SELECTED_BACKEND = 'cpu'
except Exception as _e:
    # GPU 版本不可用时回退到 CPU
    from PyLoRa import PyLoRa as _PyLoRaSelected
    SELECTED_BACKEND = 'cpu'
    print(f"[test_claude] GPU backend unavailable, fallback to CPU: {_e}")
PyLoRa = _PyLoRaSelected
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pytest
import json
from datetime import datetime
import random
from scipy.ndimage import gaussian_filter1d

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

# 对应 SF=11/12 且 BW=125 kHz 时，LoRa 会开启低速率模式，需要忽略符号的低两位
def normalize_symbol(symbol, sf, bw):
    arr = np.asarray(symbol, dtype=np.int64)
    if sf in (11, 12) and abs(bw - 125e3) < 1e-6:
        arr = arr >> 2
    if np.isscalar(symbol):
        return int(arr)
    return arr

def load_sig(file_path):
    return lora.read_file(file_path)

@pytest.mark.parametrize(
    "data, sf,snr_min,snr_max,step,epochs",
    [
        # ("mock", 7, -40, -2, 2, 1),
        # ("mock",8,-40,-2,3,4),
        # ("mock",9,-40,-2,3,2),
        ("mock",7,-40,-2,3,1)
    ]
)
def test_multiple_snr(data, sf,snr_min,snr_max,step,epochs):
    lora.sf = sf
    # SNR范围从-30到-3，步进为3
    snr_range = np.arange(snr_min, snr_max, step)
    # epochs = 1  # 可以根据需要调整
    # 存储每个函数在不同SNR下的准确率
    results = {
        'ChirpSmoother': [],
        'LoRa Trimmer': [],
        'LoRaPHY-CPA': [],
        'LoRaPHY-FPA':[],
        'MFFT':[],
        'HFFT':[],

    }
    
    ideal_path = os.path.join("./datasets",data,str(sf),"our")
    tradition_path = os.path.join("./datasets",data,str(sf),"tradition")
    # 测试函数配置
    test_configs = [
        ('ChirpSmoother', ideal_path, lora.our_ideal_decode_decodev2),
        ('HFFT', tradition_path ,lora.hfft_decode),
        ('MFFT', tradition_path ,lora.MFFT),
        ('LoRa Trimmer', tradition_path, lora.loratrimmer_decode),
        ('LoRaPHY-CPA', tradition_path, lora.loraphy),
        ('LoRaPHY-FPA',tradition_path, lora.loraphy_fpa),

    ]
    
    # 时间估算变量
    start_time = time.time()
    total_snr_count = len(snr_range)
    total_tests = len(test_configs)
    total_iterations = total_snr_count * total_tests * (2 ** sf) * epochs
    completed_iterations = 0
    
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    # 对每个SNR值进行测试
    for snr_idx, snr in enumerate(snr_range):
        print(f"\n=== Testing SNR: {snr} ({snr_idx + 1}/{total_snr_count}) ===")
        # 测试每个函数
        for test_idx, (name, dir_path, func) in enumerate(test_configs):
            result = 0
            # GPU批处理路径：当后端为GPU且提供批量接口时，一次性加载并处理一批
            is_gpu = SELECTED_BACKEND == 'gpu'
            can_batch = is_gpu and hasattr(lora, 'batch_our_ideal_decode_decodev2')
            cp = None
            if is_gpu:
                try:
                    import cupy as cp  # noqa: F401
                except Exception:
                    cp = None
            # 预读所有符号，构建 [B,T] 数组，B=2**sf
            all_sigs = []
            truths = []
            if can_batch:
                for i in range(2 ** sf):
                    file_path = os.path.join(dir_path, str(i) + ".cfile")
                    sig = load_sig(file_path)
                    all_sigs.append(sig)
                    truths.append(i)
                all_sigs = np.stack(all_sigs, axis=0)
                truths_eval = normalize_symbol(truths, sf, lora.bw)

            for epoch in range(epochs):
                if can_batch:
                    # 批量加噪（一次生成整批噪声，再分批解码）
                    noisy_all = lora.add_noise_batch(all_sigs, snr=snr) if hasattr(lora, 'add_noise_batch') else np.stack([lora.add_noise(sig=s, snr=snr) for s in all_sigs], axis=0)
                    B = 2 ** sf
                    # 方法相关的安全 batch size，避免 OOM
                    if name in ('LoRaPHY-CPA', 'LoRaPHY-FPA'):
                        batch_size = 64
                    else:
                        batch_size = 256
                    # 分批推理
                    all_rets = []
                    for start in range(0, B, batch_size):
                        end = min(start + batch_size, B)
                        noisy = noisy_all[start:end]
                        if name == 'ChirpSmoother' and hasattr(lora, 'batch_our_ideal_decode_decodev2'):
                            rets, _ = lora.batch_our_ideal_decode_decodev2(noisy)
                        elif name == 'HFFT' and hasattr(lora, 'batch_hfft_decode'):
                            rets, _ = lora.batch_hfft_decode(noisy)
                        elif name == 'MFFT' and hasattr(lora, 'batch_MFFT'):
                            rets, _ = lora.batch_MFFT(noisy)
                        elif name == 'LoRa Trimmer' and hasattr(lora, 'batch_loratrimmer_decode'):
                            rets, _ = lora.batch_loratrimmer_decode(noisy)
                        elif name == 'LoRaPHY-CPA' and hasattr(lora, 'batch_loraphy'):
                            rets, _ = lora.batch_loraphy(noisy)
                        elif name == 'LoRaPHY-FPA' and hasattr(lora, 'batch_loraphy_fpa'):
                            rets, _ = lora.batch_loraphy_fpa(noisy)
                        else:
                            rets = np.array([func(sig=x)[0] for x in noisy])
                        all_rets.append(rets)
                        # 主动释放显存
                        if cp is not None:
                            try:
                                cp.cuda.Device().synchronize()
                                cp.get_default_memory_pool().free_all_blocks()
                            except Exception:
                                pass
                    rets = np.concatenate(all_rets, axis=0)
                    rets_eval = normalize_symbol(rets, sf, lora.bw)
                    result += int(np.sum(rets_eval == truths_eval))
                    completed_iterations += B
                else:
                    for i in range(2 ** sf):
                        file_path = os.path.join(dir_path, str(i) + ".cfile")
                        truth = i
                        sig = load_sig(file_path)
                        chirp = lora.add_noise(sig=sig, snr=snr)
                        ret = func(sig=chirp)[0]
                        truth_eval = normalize_symbol(truth, sf, lora.bw)
                        ret_eval = normalize_symbol(ret, sf, lora.bw)
                        if ret_eval == truth_eval:
                            result += 1
                        completed_iterations += 1

                # 进度输出
                elapsed_time = time.time() - start_time
                if completed_iterations > 0:
                    avg_time_per_iteration = elapsed_time / completed_iterations
                    remaining_iterations = total_iterations - completed_iterations
                    estimated_remaining_time = remaining_iterations * avg_time_per_iteration
                    progress_percent = (completed_iterations / total_iterations) * 100
                    print(f"  Progress: {progress_percent:.1f}% | Elapsed: {format_time(elapsed_time)} | Remaining: {format_time(estimated_remaining_time)}")

            accuracy = result / (epochs * 2 ** sf)
            results[name].append(accuracy)
            print(f"  {name}: {accuracy:.4f}")
    
    # 完成时间统计
    total_elapsed_time = time.time() - start_time
    print(f"\n=== Test Completed ===")
    print(f"Total time elapsed: {format_time(total_elapsed_time)}")
    print(f"Total iterations: {total_iterations}")
    print(f"Average time per iteration: {total_elapsed_time/total_iterations:.4f}s")
    
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
    plt.figure(figsize=(16, 12))
    
    # 定义不同线型、颜色和标记点
    styles = [
        ('ChirpSmoother', 'red', 'solid', '*'),      # 五角星
        ('HFFT', '#1f77b4', 'solid', '^'),          # 正三角
        ('LoRa Trimmer', '#ff7f0e', 'dashed', 'v'),  # 倒三角
        ('LoRaPHY-CPA', '#2ca02c', 'dotted', 'x'),   # 叉号
        ('LoRaPHY-FPA', '#d62728', 'dashdot', '|'),  # 竖线
        ('MFFT', '#9467bd', 'solid', 'd')           # 菱形
    ]
    
    # 使用不同的线型绘制每个方法的结果，添加平滑效果和标记点
    for name, color, linestyle, marker in styles:
        if name in results:
            # 应用高斯滤波平滑
            smoothed_results = gaussian_filter1d(results[name], sigma=0.5)
            # ChirpSmoother使用大小20的五角星，其他方法使用大小16的标记
            markersize = 24 if name == 'ChirpSmoother' else 16
            plt.plot(snr_range, smoothed_results, color=color, linestyle=linestyle, 
                    label=name, alpha=0.8, linewidth=2, marker=marker, markersize=markersize)

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    # plt.title(f'Decoder Performance vs SNR(SF={sf}) mock data')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, columnspacing=4, handletextpad=2, handlelength=4,labelspacing=2)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    s1 = Singleton()
    folder_name = s1.get_folder()
    plt.savefig(os.path.join(folder_name,f'decoder_performance_SF{sf}'  + '.png'), dpi=300, bbox_inches='tight')
    plt.close()

def draw_from_json(json_path, output_path=None):
    """
    从JSON文件读取数据并绘制图表
    
    Args:
        json_path: JSON文件路径
        output_path: 输出图片路径，如果不指定则保存到JSON文件同目录
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # 提取数据
    results = {k: v for k, v in data.items() if k not in ['snr_range', 'sf']}
    snr_range = data['snr_range']
    sf = data['sf']
    
    plt.figure(figsize=(16, 12))
    
    # 定义不同线型、颜色和标记点（与draw函数一致）
    styles = [
        ('ChirpSmoother', 'red', 'solid', '*'),  # 五角星
        ('HFFT', '#1f77b4', 'solid', '^'),  # 正三角
        ('LoRa Trimmer', '#ff7f0e', 'dashed', 'v'),  # 倒三角
        ('LoRaPHY-CPA', '#2ca02c', 'dotted', 'x'),  # 叉号
        ('LoRaPHY-FPA', '#d62728', 'dashdot', '|'),  # 竖线
        ('MFFT', '#9467bd', 'solid', 'd')  # 菱形
    ]
    
    # 使用不同的线型绘制每个方法的结果，添加平滑效果和标记点
    for name, color, linestyle, marker in styles:
        if name in results:
            # 应用高斯滤波平滑
            smoothed_results = gaussian_filter1d(results[name], sigma=0.5)
            # ChirpSmoother使用大小20的五角星，其他方法使用大小16的标记
            markersize = 24 if name == 'ChirpSmoother' else 16
            plt.plot(snr_range, smoothed_results, color=color, linestyle=linestyle, 
                    label=name, alpha=0.8, linewidth=2, marker=marker, markersize=markersize)

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    # plt.title(f'Decoder Performance vs SNR(SF={sf}) mock data')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, columnspacing=4, handletextpad=2, handlelength=4,labelspacing=2)
    
    # 确定输出路径
    if output_path is None:
        # 如果没有指定输出路径，保存到JSON文件同目录
        import os
        json_dir = os.path.dirname(json_path)
        output_path = os.path.join(json_dir, f'decoder_performance_SF{sf}_from_json.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存到: {output_path}")
    return output_path

if __name__ == '__main__':
    draw_from_json("./output/SF8/sf8.json")
    draw_from_json("./output/SF9/sf9.json")
    draw_from_json("./output/SF10/sf10.json")
    draw_from_json("./output/SF11/sf11.json")
    draw_from_json("./output/SF12/sf12.json")
    # test_multiple_snr  ("mock", 10,-40,-2,3,32)
