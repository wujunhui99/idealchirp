import sys

# 动态选择 CPU/GPU 版本的 PyLoRa，默认 CPU，命令行包含 'gpu' 时启用 GPU 版本
_use_gpu = True
try:
    if _use_gpu:
        from PyLoRa_GPU import PyLoRa as _PyLoRaSelected
        SELECTED_BACKEND = "gpu"
    else:
        from PyLoRa import PyLoRa as _PyLoRaSelected
        SELECTED_BACKEND = "cpu"
except Exception as _e:
    from PyLoRa import PyLoRa as _PyLoRaSelected
    SELECTED_BACKEND = "cpu"
    print(f"[paral_ber_codex] GPU backend unavailable, fallback to CPU: {_e}")

PyLoRa = _PyLoRaSelected

import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d


class Singleton:
    _instance = None
    _folder_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y%m%d%H%M%S")
            base_dir = "./BER"
            os.makedirs(base_dir, exist_ok=True)
            folder_name = os.path.join(base_dir, f"record{formatted_time}")
            os.makedirs(folder_name)
            cls._folder_name = folder_name
        return cls._instance

    def get_folder(self):
        return self._folder_name


lora = PyLoRa()


def load_sig(file_path):
    return lora.read_file(file_path)


def normalize_symbol(symbol, sf, bw):
    """对应 SF=11/12 且 BW=125 kHz 时，LoRa 会开启低速率模式，需要忽略符号的低两位"""
    arr = np.asarray(symbol, dtype=np.int64)
    if sf in (11, 12) and abs(bw - 125e3) < 1e-6:
        arr = arr >> 2
    if np.isscalar(symbol):
        return int(arr)
    return arr


def effective_bit_width(sf, bw):
    """返回用于 BER 计算的有效比特数"""
    if sf in (11, 12) and abs(bw - 125e3) < 1e-6:
        return sf - 2
    return sf


def count_bit_errors(pred, truth):
    """计算两个符号间的比特错误数"""
    return int(int(pred ^ truth).bit_count())


bit_counter = np.vectorize(lambda v: int(v).bit_count(), otypes=[np.int64])


@pytest.mark.parametrize(
    "data, sf, snr_min, snr_max, step, epochs",
    [
        # ("mock", 7, -40, -2, 1, 32),
        # ("mock", 8, -40, -2, 1, 16),
        # ("mock", 9, -40, -2, 1, 8),
        # ("mock", 10, -40, -2, 1, 4),
        ("mock", 11, -40, -2, 1, 2),
        ("mock", 12, -40, -2, 1, 1),
    ],
)
def test_multiple_snr(data, sf, snr_min, snr_max, step, epochs):
    lora.sf = sf
    bit_width = effective_bit_width(sf, lora.bw)
    snr_range = np.arange(snr_min, snr_max, step)

    results = {
        "ChirpSmoother": [],
        "LoRa Trimmer": [],
        "LoRaPHY-CPA": [],
        "LoRaPHY-FPA": [],
        "MFFT": [],
        "HFFT": [],
    }

    ideal_path = os.path.join("./datasets", data, str(sf), "our")
    tradition_path = os.path.join("./datasets", data, str(sf), "tradition")
    test_configs = [
        ("ChirpSmoother", ideal_path, lora.our_ideal_decode_decodev2),
        ("HFFT", tradition_path, lora.hfft_decode),
        ("MFFT", tradition_path, lora.MFFT),
        ("LoRa Trimmer", tradition_path, lora.loratrimmer_decode),
        ("LoRaPHY-CPA", tradition_path, lora.loraphy),
        ("LoRaPHY-FPA", tradition_path, lora.loraphy_fpa),
    ]

    truths_all = np.arange(2**sf, dtype=np.int64)
    truths_eval_all = normalize_symbol(truths_all, sf, lora.bw)

    start_time = time.time()
    total_snr_count = len(snr_range)
    total_tests = len(test_configs)
    total_iterations = total_snr_count * total_tests * (2**sf) * epochs
    completed_iterations = 0

    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    is_gpu = SELECTED_BACKEND == "gpu"
    try:
        cp = __import__("cupy") if is_gpu else None
    except Exception:
        cp = None

    def _gen_noisy(all_sigs_arr, snr_slice_arr):
        """Generate noisy signals; compatible with old add_noise_batch signatures."""
        add_noise_fn = getattr(lora, "add_noise_batch", None)
        if add_noise_fn is None:
            snr_val = float(np.asarray(snr_slice_arr).reshape(-1)[0])
            return np.stack([lora.add_noise(sig=s, snr=snr_val) for s in all_sigs_arr], axis=0)
        try:
            return add_noise_fn(all_sigs_arr, snr=snr_slice_arr, return_numpy=False)
        except TypeError:
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
        bit_errors = np.zeros(total_snr_count, dtype=np.int64) if is_gpu else None
        can_batch = is_gpu and hasattr(lora, "batch_our_ideal_decode_decodev2")

        all_sigs = []
        if can_batch:
            for i in range(2**sf):
                file_path = os.path.join(dir_path, str(i) + ".cfile")
                sig = load_sig(file_path)
                all_sigs.append(sig)
            all_sigs = np.stack(all_sigs, axis=0)
            truths_eval = truths_eval_all

            B = 2**sf
            gpu_chunk_limit_bytes = int(os.getenv("GPU_SNR_CHUNK_BYTES", 8 * 1024 * 1024 * 1024))
            per_snr_bytes = all_sigs.nbytes
            snr_chunk = max(1, min(total_snr_count, int(gpu_chunk_limit_bytes // per_snr_bytes) if per_snr_bytes else total_snr_count))
            snr_chunk = max(1, snr_chunk)

            def get_batch_size(method_name):
                if method_name in ("LoRaPHY-CPA", "LoRaPHY-FPA"):
                    return 256
                return 1024

            for _ in range(epochs):
                for snr_start in range(0, total_snr_count, snr_chunk):
                    snr_slice = snr_range[snr_start : snr_start + snr_chunk]
                    noisy_all = _gen_noisy(all_sigs, snr_slice)
                    noisy_flat = noisy_all.reshape(-1, noisy_all.shape[-1])

                    batch_size = get_batch_size(name)
                    all_rets = []
                    total_flat = int(noisy_flat.shape[0])
                    for start in range(0, total_flat, batch_size):
                        end = min(start + batch_size, total_flat)
                        noisy = noisy_flat[start:end]
                        if name == "ChirpSmoother" and hasattr(lora, "batch_our_ideal_decode_decodev2"):
                            rets, _ = lora.batch_our_ideal_decode_decodev2(noisy)
                        elif name == "HFFT" and hasattr(lora, "batch_hfft_decode"):
                            rets, _ = lora.batch_hfft_decode(noisy)
                        elif name == "MFFT" and hasattr(lora, "batch_MFFT"):
                            rets, _ = lora.batch_MFFT(noisy)
                        elif name == "LoRa Trimmer" and hasattr(lora, "batch_loratrimmer_decode"):
                            rets, _ = lora.batch_loratrimmer_decode(noisy)
                        elif name == "LoRaPHY-CPA" and hasattr(lora, "batch_loraphy"):
                            rets, _ = lora.batch_loraphy(noisy)
                        elif name == "LoRaPHY-FPA" and hasattr(lora, "batch_loraphy_fpa"):
                            rets, _ = lora.batch_loraphy_fpa(noisy)
                        else:
                            rets = np.array([func(sig=x)[0] for x in noisy])
                        all_rets.append(rets)
                        if cp is not None:
                            try:
                                cp.cuda.Device().synchronize()
                            except Exception:
                                pass

                    rets = np.concatenate(all_rets, axis=0)
                    rets_eval = normalize_symbol(np.asarray(rets), sf, lora.bw).reshape(len(snr_slice), B)
                    xor_vals = np.bitwise_xor(rets_eval.astype(np.int64), truths_eval.astype(np.int64))
                    bit_counts = bit_counter(xor_vals)
                    for local_idx in range(len(snr_slice)):
                        bit_errors[snr_start + local_idx] += int(np.sum(bit_counts[local_idx]))
                    completed_iterations += len(snr_slice) * B

                    elapsed_time = time.time() - start_time
                    avg_time = elapsed_time / completed_iterations if completed_iterations else 0
                    remaining = total_iterations - completed_iterations
                    eta = remaining * avg_time if avg_time else 0
                    progress_percent = (completed_iterations / total_iterations) * 100
                    print(
                        f"  Progress: {progress_percent:.1f}% | Elapsed: {format_time(elapsed_time)} | Remaining: {format_time(eta)}"
                    )

            total_bits = epochs * (2**sf) * bit_width
            accuracy = bit_errors / total_bits if total_bits > 0 else np.zeros_like(bit_errors, dtype=np.float64)
            results[name].extend(accuracy.tolist())
            for snr_idx, ber in enumerate(accuracy):
                print(f"  SNR {snr_range[snr_idx]}: BER={ber:.6f}")
        else:
            for snr_idx, snr in enumerate(snr_range):
                bit_error_sum = 0
                for _ in range(epochs):
                    for i in range(2**sf):
                        file_path = os.path.join(dir_path, str(i) + ".cfile")
                        truth_eval = int(truths_eval_all[i])
                        sig = load_sig(file_path)
                        chirp = lora.add_noise(sig=sig, snr=snr)
                        ret = func(sig=chirp)[0]
                        ret_eval = normalize_symbol(ret, sf, lora.bw)
                        bit_error_sum += count_bit_errors(int(ret_eval), int(truth_eval))
                        completed_iterations += 1

                    elapsed_time = time.time() - start_time
                    if completed_iterations > 0:
                        avg_time = elapsed_time / completed_iterations
                        remaining = total_iterations - completed_iterations
                        eta = remaining * avg_time
                        progress_percent = (completed_iterations / total_iterations) * 100
                        print(
                            f"  Progress: {progress_percent:.1f}% | Elapsed: {format_time(elapsed_time)} | Remaining: {format_time(eta)}"
                        )

                total_bits = epochs * (2**sf) * bit_width
                ber = bit_error_sum / total_bits if total_bits > 0 else 0.0
                results[name].append(ber)
                print(f"  {name} @ SNR {snr}: BER={ber:.6f}")

    total_elapsed_time = time.time() - start_time
    print("\n=== Test Completed ===")
    print(f"Total time elapsed: {format_time(total_elapsed_time)}")
    print(f"Total iterations: {total_iterations}")
    print(f"Average time per iteration: {total_elapsed_time / total_iterations:.4f}s")

    s1 = Singleton()
    folder_name = s1.get_folder()
    draw(results, snr_range, sf, folder_name)
    with open(os.path.join(folder_name, f"./sf{sf}.json"), "w") as file:
        res = results.copy()
        res["snr_range"] = snr_range.tolist()
        res["sf"] = sf
        json.dump(res, file, indent=4)
    return results, snr_range, results


def draw(results, snr_range, sf, folder_name):
    plt.figure(figsize=(16, 12))
    styles = [
        ("ChirpSmoother", "red", "solid", "*"),
        ("HFFT", "#1f77b4", "solid", "^"),
        ("LoRa Trimmer", "#ff7f0e", "dashed", "v"),
        ("LoRaPHY-CPA", "#2ca02c", "dotted", "x"),
        ("LoRaPHY-FPA", "#d62728", "dashdot", "|"),
        ("MFFT", "#9467bd", "solid", "d"),
    ]

    for name, color, linestyle, marker in styles:
        if name in results:
            smoothed_results = gaussian_filter1d(results[name], sigma=0.5)
            markersize = 24 if name == "ChirpSmoother" else 16
            plt.plot(
                snr_range,
                smoothed_results,
                color=color,
                linestyle=linestyle,
                label=name,
                alpha=0.8,
                linewidth=2,
                marker=marker,
                markersize=markersize,
            )

    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("BER", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10, columnspacing=4, handletextpad=2, handlelength=4, labelspacing=2)
    s1 = Singleton()
    folder_name = s1.get_folder()
    plt.savefig(
        os.path.join(folder_name, f"ber_performance_SF{sf}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def draw_from_json(json_path, output_path=None):
    """从 JSON 文件读取数据并绘制 BER 图"""
    with open(json_path, "r") as file:
        data = json.load(file)

    results = {k: v for k, v in data.items() if k not in ["snr_range", "sf"]}
    snr_range = data["snr_range"]
    sf = data["sf"]

    plt.figure(figsize=(16, 12))
    styles = [
        ("ChirpSmoother", "red", "solid", "*"),
        ("HFFT", "#1f77b4", "solid", "^"),
        ("LoRa Trimmer", "#ff7f0e", "dashed", "v"),
        ("LoRaPHY-CPA", "#2ca02c", "dotted", "x"),
        ("LoRaPHY-FPA", "#d62728", "dashdot", "|"),
        ("MFFT", "#9467bd", "solid", "d"),
    ]

    for name, color, linestyle, marker in styles:
        if name in results:
            smoothed_results = gaussian_filter1d(results[name], sigma=0.5)
            markersize = 24 if name == "ChirpSmoother" else 16
            plt.plot(
                snr_range,
                smoothed_results,
                color=color,
                linestyle=linestyle,
                label=name,
                alpha=0.8,
                linewidth=2,
                marker=marker,
                markersize=markersize,
            )

    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("BER", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10, columnspacing=4, handletextpad=2, handlelength=4, labelspacing=2)

    if output_path is None:
        json_dir = os.path.dirname(json_path)
        output_path = os.path.join(json_dir, f"ber_performance_SF{sf}_from_json.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图表已保存到: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Run via pytest to generate BER results, or call draw_from_json(json_path).")
