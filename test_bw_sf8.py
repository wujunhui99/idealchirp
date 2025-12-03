import os
import time
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from PyLoRa import PyLoRa


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def run_eval_for_bw(sf: int, bw: float, snr_min: int, snr_max: int, step: int, epochs: int = 1):
    lora = PyLoRa()
    lora.sf = sf
    lora.bw = bw
    # Keep fs=1e6; OSR will be 4 for 250k, 2 for 500k (integers)
    lora.fs = 1_000_000
    # Prepare constants for LoRa Trimmer
    lora.dataE1, lora.dataE2 = lora.gen_constants()

    snr_range = np.arange(snr_min, snr_max, step)

    # Methods to test (same names and functions as test_claude)
    test_configs = [
        ("ChirpSmoother", lora.our_ideal_decode_decodev2),
        ("HFFT", lora.hfft_decode),
        ("MFFT", lora.MFFT),
        ("LoRa Trimmer", lora.loratrimmer_decode),
        ("LoRaPHY-CPA", lora.loraphy),
        ("LoRaPHY-FPA", lora.loraphy_fpa),
    ]

    results = {name: [] for name, _ in test_configs}

    start_time = time.time()
    total_snr_count = len(snr_range)
    total_tests = len(test_configs)
    total_iterations = total_snr_count * total_tests * (2 ** sf) * epochs
    completed_iterations = 0

    for snr_idx, snr in enumerate(snr_range):
        print(f"\n=== [BW={int(bw/1e3)}kHz, SF={sf}] Testing SNR: {snr} ({snr_idx + 1}/{total_snr_count}) ===")
        for name, func in test_configs:
            correct = 0
            for _ in range(epochs):
                for truth in range(2 ** sf):
                    # ChirpSmoother uses ideal_chirp; others use real_chirp
                    if name == "ChirpSmoother":
                        chirp = lora.ideal_chirp(f0=truth, iq_invert=0)
                    else:
                        chirp = lora.real_chirp(f0=truth, iq_invert=0)
                    noisy = lora.add_noise(sig=chirp, snr=snr)
                    pred, _ = func(sig=noisy)
                    if pred == truth:
                        correct += 1
                    completed_iterations += 1

            accuracy = correct / (epochs * (2 ** sf))
            results[name].append(accuracy)
            print(f"  {name}: {accuracy:.4f}")

        # progress
        elapsed_time = time.time() - start_time
        if completed_iterations > 0:
            avg_time = elapsed_time / completed_iterations
            remaining = max(total_iterations - completed_iterations, 0)
            eta = remaining * avg_time
            pct = (completed_iterations / total_iterations) * 100
            print(f"  Progress: {pct:.1f}% | Elapsed: {format_time(elapsed_time)} | Remaining: {format_time(eta)}")

    return results, snr_range.tolist()


def draw_results(results: dict, snr_range, sf: int, bw_khz: int, output_dir: str) -> str:
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
            smoothed = gaussian_filter1d(results[name], sigma=0.5)
            markersize = 24 if name == "ChirpSmoother" else 16
            plt.plot(
                snr_range,
                smoothed,
                color=color,
                linestyle=linestyle,
                label=name,
                alpha=0.8,
                linewidth=2,
                marker=marker,
                markersize=markersize,
            )

    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10, columnspacing=4, handletextpad=2, handlelength=4, labelspacing=2)

    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, f"decoder_performance_SF{sf}_BW{bw_khz}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")
    return out_path


def main():
    sf = 8
    snr_min, snr_max, step = -40, -2, 3
    epochs = 1

    output_dir = os.path.join("./output", f"SF{sf}")
    ensure_dir(output_dir)

    # BW: 250 kHz
    bw_250 = 250_000.0
    results_250, snr_range_250 = run_eval_for_bw(sf=sf, bw=bw_250, snr_min=snr_min, snr_max=snr_max, step=step, epochs=epochs)
    json_250 = {
        **results_250,
        "snr_range": snr_range_250,
        "sf": sf,
        "bw": int(bw_250),
    }
    json_250_path = os.path.join(output_dir, f"sf{sf}_bw250.json")
    with open(json_250_path, "w") as f:
        json.dump(json_250, f, indent=4)
    print(f"Saved JSON: {json_250_path}")
    draw_results(results_250, snr_range_250, sf=sf, bw_khz=250, output_dir=output_dir)

    # BW: 500 kHz
    bw_500 = 500_000.0
    results_500, snr_range_500 = run_eval_for_bw(sf=sf, bw=bw_500, snr_min=snr_min, snr_max=snr_max, step=step, epochs=epochs)
    json_500 = {
        **results_500,
        "snr_range": snr_range_500,
        "sf": sf,
        "bw": int(bw_500),
    }
    json_500_path = os.path.join(output_dir, f"sf{sf}_bw500.json")
    with open(json_500_path, "w") as f:
        json.dump(json_500, f, indent=4)
    print(f"Saved JSON: {json_500_path}")
    draw_results(results_500, snr_range_500, sf=sf, bw_khz=500, output_dir=output_dir)


if __name__ == "__main__":
    main()


