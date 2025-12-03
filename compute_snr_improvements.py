import json
import os
import argparse
from typing import Dict, List, Optional, Tuple


def find_snr_threshold(snr_values: List[float], success_rates: List[float], threshold: float = 0.9) -> float:
    """Return the SNR at which success rate first reaches threshold using linear interpolation.

    If the first point already meets threshold, return the first SNR. If never reaches, return last SNR.
    """
    for i, rate in enumerate(success_rates):
        if rate >= threshold:
            if i == 0:
                return float(snr_values[0])
            prev_snr = snr_values[i - 1]
            curr_snr = snr_values[i]
            prev_rate = success_rates[i - 1]
            curr_rate = success_rates[i]
            # Linear interpolation
            return float(prev_snr + (threshold - prev_rate) * (curr_snr - prev_snr) / (curr_rate - prev_rate))
    return float(snr_values[-1])


def load_sf_data(sf: int, base_dir: str = "output") -> Optional[Dict]:
    """Load JSON data for a given SF from base_dir/SF{sf}/sf{sf}.json."""
    file_path = os.path.join(base_dir, f"SF{sf}", f"sf{sf}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return json.load(f)


def compute_thresholds(data: Dict, methods: List[str], threshold: float) -> Dict[str, float]:
    snr_range = data["snr_range"]
    result: Dict[str, float] = {}
    for method in methods:
        if method in data:
            result[method] = find_snr_threshold(snr_range, data[method], threshold)
    return result


def compute_improvements_for_sf(sf: int, thresholds: Dict[str, float]) -> Tuple[float, float, float, Tuple[str, float]]:
    """Return (cs_snr, improvement_vs_trimmer, improvement_vs_hfft, (best_baseline_name, best_delta))."""
    cs = thresholds["ChirpSmoother"]
    imp_vs_tr = thresholds.get("LoRa Trimmer", float("nan")) - cs
    imp_vs_hfft = thresholds.get("HFFT", float("nan")) - cs
    # Best baseline vs CS (positive means CS requires lower SNR -> improvement)
    best_baseline: Tuple[str, float] = max(
        ((m, thresholds[m] - cs) for m in thresholds.keys() if m != "ChirpSmoother"),
        key=lambda p: p[1],
    )
    return cs, imp_vs_tr, imp_vs_hfft, best_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute 90% SNR threshold improvements for ChirpSmoother")
    parser.add_argument("--base-dir", default="output", help="Base directory containing SF folders (default: output)")
    parser.add_argument("--threshold", type=float, default=0.9, help="Success rate threshold (default: 0.9)")
    parser.add_argument("--sfs", type=int, nargs="*", default=[7, 8, 9, 10, 11, 12], help="SF list to include")
    args = parser.parse_args()

    methods = [
        "ChirpSmoother",
        "LoRa Trimmer",
        "LoRaPHY-CPA",
        "LoRaPHY-FPA",
        "MFFT",
        "HFFT",
    ]

    header = (
        "SF",
        "CS_Thr(dB)",
        "Δ vs LoRaTrimmer(dB)",
        "Δ vs HFFT(dB)",
        "Δ vs LoRaPHY-CPA(dB)",
        "Max Δ vs others(dB)",
        "Best baseline",
    )
    print(
        f"{header[0]:>3}  {header[1]:>11}  {header[2]:>20}  {header[3]:>15}  {header[4]:>22}  {header[5]:>20}  {header[6]}"
    )
    print("-" * 100)

    for sf in args.sfs:
        data = load_sf_data(sf, args.base_dir)
        if data is None:
            print(f"SF{sf:>2}: data not found under {args.base_dir}/SF{sf}/sf{sf}.json")
            continue
        thresholds = compute_thresholds(data, methods, args.threshold)
        if "ChirpSmoother" not in thresholds:
            print(f"SF{sf:>2}: ChirpSmoother data missing")
            continue

        cs, imp_vs_tr, imp_vs_hfft, (best_name, best_delta) = compute_improvements_for_sf(sf, thresholds)
        imp_vs_cpa = thresholds.get("LoRaPHY-CPA", float("nan")) - cs
        print(
            f"{sf:>3}  {cs:11.3f}  {imp_vs_tr:20.3f}  {imp_vs_hfft:15.3f}  {imp_vs_cpa:22.3f}  {best_delta:20.3f}  {best_name}"
        )


if __name__ == "__main__":
    main()


