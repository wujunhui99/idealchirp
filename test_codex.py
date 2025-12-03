import sys
# 动态选择 CPU/GPU 版本的 PyLoRa，默认 CPU，命令行包含 'gpu' 时启用 GPU 版本
_use_gpu = True
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
        ("mock", 7, -40, -2, 1, 32),
        ("mock",8,-40,-2,1,16),
        ("mock",9,-40,-2,1,8),
        ("mock",10,-40,-2,1,4),
        ("mock",11,-40,-2,1,2),
        ("mock",12,-40,-2,1,1),
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

    # 预读所有符号，构建 [B,T] 数组，B=2**sf（GPU 批处理一次读完，CPU 保持旧逻辑）
    is_gpu = SELECTED_BACKEND == 'gpu'
    try:
        cp = __import__('cupy') if is_gpu else None
    except Exception:
        cp = None

    def _gen_noisy(all_sigs_arr, snr_slice_arr):
        """Generate noisy signals; compatible with old add_noise_batch signatures."""
        add_noise_fn = getattr(lora, 'add_noise_batch', None)
        if add_noise_fn is None:
            return np.stack([lora.add_noise(sig=s, snr=float(snr_slice_arr)) for s in all_sigs_arr], axis=0)
        try:
            return add_noise_fn(all_sigs_arr, snr=snr_slice_arr, return_numpy=False)
        except TypeError:
            # Fallback for legacy signature; if SNR is vector, call per-snr to avoid math.pow errors
            try:
                noisy = add_noise_fn(all_sigs_arr, snr=snr_slice_arr)
                if cp is not None:
                    try:
                        noisy = cp.asarray(noisy)
                    except Exception:
                        pass
                return noisy
            except Exception:
                snr_flat = np.asarray(snr_slice_arr).reshape(-1)
                outs = []
                for snr_scalar in snr_flat:
                    out = add_noise_fn(all_sigs_arr, snr=float(snr_scalar))
                    outs.append(cp.asarray(out) if cp is not None else np.asarray(out))
                return cp.stack(outs, axis=0) if cp is not None else np.stack(outs, axis=0)

    for test_idx, (name, dir_path, func) in enumerate(test_configs):
        print(f"\n=== Testing {name} ({test_idx + 1}/{total_tests}) ===")
        result = np.zeros(total_snr_count, dtype=np.int64) if is_gpu else 0
        can_batch = is_gpu and hasattr(lora, 'batch_our_ideal_decode_decodev2')

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
            truths_eval = np.asarray(truths_eval, dtype=np.int64)

        # GPU并行路径：一次处理多个SNR，显著提升吞吐
        if can_batch:
            B = 2 ** sf
            # 动态控制 SNR 维度的 chunk，避免占用过多显存。默认上限 512MB。
            gpu_chunk_limit_bytes = int(os.getenv("GPU_SNR_CHUNK_BYTES", 512 * 1024 * 1024))
            per_snr_bytes = all_sigs.nbytes
            snr_chunk = max(1, min(total_snr_count, int(gpu_chunk_limit_bytes // per_snr_bytes) if per_snr_bytes else total_snr_count))
            snr_chunk = max(1, snr_chunk)

            # 方法相关的批大小：FPA/CPA 稍小，其余更大以提升占用
            def get_batch_size(method_name):
                if method_name in ('LoRaPHY-CPA', 'LoRaPHY-FPA'):
                    return 256
                return 1024

            for epoch in range(epochs):
                for snr_start in range(0, total_snr_count, snr_chunk):
                    snr_slice = snr_range[snr_start:snr_start + snr_chunk]
                    # 直接在 GPU 上生成带噪声的 [S,B,T]
                    noisy_all = _gen_noisy(all_sigs, snr_slice)
                    noisy_flat = noisy_all.reshape(-1, noisy_all.shape[-1])

                    batch_size = get_batch_size(name)
                    all_rets = []
                    for start in range(0, noisy_flat.shape[0], batch_size):
                        end = min(start + batch_size, noisy_flat.shape[0])
                        noisy = noisy_flat[start:end]
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
                        # 可以选择性同步显存，避免显存碎片
                        if cp is not None:
                            try:
                                cp.cuda.Device().synchronize()
                            except Exception:
                                pass

                    rets = np.concatenate(all_rets, axis=0)
                    rets_eval = normalize_symbol(np.asarray(rets), sf, lora.bw).reshape(len(snr_slice), B)
                    matches = (rets_eval == truths_eval)
                    for local_idx, snr_idx in enumerate(range(snr_start, snr_start + len(snr_slice))):
                        result[snr_idx] += int(np.sum(matches[local_idx]))
                    completed_iterations += len(snr_slice) * B

                    # 进度输出（按 chunk）
                    elapsed_time = time.time() - start_time
                    avg_time_per_iteration = elapsed_time / completed_iterations if completed_iterations else 0
                    remaining_iterations = total_iterations - completed_iterations
                    estimated_remaining_time = remaining_iterations * avg_time_per_iteration if avg_time_per_iteration else 0
                    progress_percent = (completed_iterations / total_iterations) * 100
                    print(f"  Progress: {progress_percent:.1f}% | Elapsed: {format_time(elapsed_time)} | Remaining: {format_time(estimated_remaining_time)}")

            accuracy = result / (epochs * 2 ** sf)
            results[name].extend(accuracy.tolist())
            for snr_idx, acc in enumerate(accuracy):
                print(f"  SNR {snr_range[snr_idx]}: {acc:.4f}")
        else:
            # CPU 或无批处理路径：保持原逻辑（逐 SNR）
            for snr_idx, snr in enumerate(snr_range):
                print(f"\n--- SNR: {snr} ({snr_idx + 1}/{total_snr_count}) ---")
                result_single = 0
                for epoch in range(epochs):
                    for i in range(2 ** sf):
                        file_path = os.path.join(dir_path, str(i) + ".cfile")
                        truth = i
                        sig = load_sig(file_path)
                        chirp = lora.add_noise(sig=sig, snr=snr)
                        ret = func(sig=chirp)[0]
                        truth_eval = normalize_symbol(truth, sf, lora.bw)
                        ret_eval = normalize_symbol(ret, sf, lora.bw)
                        if ret_eval == truth_eval:
                            result_single += 1
                        completed_iterations += 1

                    # 进度输出
                    elapsed_time = time.time() - start_time
                    if completed_iterations > 0:
                        avg_time_per_iteration = elapsed_time / completed_iterations
                        remaining_iterations = total_iterations - completed_iterations
                        estimated_remaining_time = remaining_iterations * avg_time_per_iteration
                        progress_percent = (completed_iterations / total_iterations) * 100
                        print(f"  Progress: {progress_percent:.1f}% | Elapsed: {format_time(elapsed_time)} | Remaining: {format_time(estimated_remaining_time)}")

                accuracy = result_single / (epochs * 2 ** sf)
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
