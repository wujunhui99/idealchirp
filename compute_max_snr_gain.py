import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple


def load_sf_data(sf: int, base_dir: str = "output") -> Optional[Dict]:
    """Load JSON for the given SF from base_dir/SF{sf}/sf{sf}.json."""
    file_path = os.path.join(base_dir, f"SF{sf}", f"sf{sf}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return json.load(f)


def find_snr_for_rate(snr_values: Sequence[float], success_rates: Sequence[float], target: float) -> float:
    """Return interpolated SNR where success rate first reaches target."""
    for i, rate in enumerate(success_rates):
        if rate >= target:
            if i == 0:
                return float(snr_values[0])
            prev_rate = success_rates[i - 1]
            prev_snr = snr_values[i - 1]
            curr_rate = success_rates[i]
            curr_snr = snr_values[i]
            if curr_rate == prev_rate:
                return float(curr_snr)
            return float(prev_snr + (target - prev_rate) * (curr_snr - prev_snr) / (curr_rate - prev_rate))
    return float(snr_values[-1])


def compute_max_gain(
    snr_values: Sequence[float],
    cs_rates: Sequence[float],
    lt_rates: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Return (best_delta, at_accuracy, cs_snr, lt_snr) with max LT->CS SNR gap."""
    candidates = sorted({*cs_rates, *lt_rates, 0.0, 1.0})
    best_delta = float("-inf")
    best_acc = 0.0
    best_cs_snr = 0.0
    best_lt_snr = 0.0

    for acc in candidates:
        cs_snr = find_snr_for_rate(snr_values, cs_rates, acc)
        lt_snr = find_snr_for_rate(snr_values, lt_rates, acc)
        delta = lt_snr - cs_snr
        if delta > best_delta:
            best_delta = delta
            best_acc = acc
            best_cs_snr = cs_snr
            best_lt_snr = lt_snr
    return best_delta, best_acc, best_cs_snr, best_lt_snr


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find the maximum SNR improvement of ChirpSmoother over LoRa Trimmer "
            "per SF across all accuracies."
        )
    )
    parser.add_argument("--base-dir", default="output", help="Directory containing SF folders (default: output)")
    parser.add_argument("--sfs", type=int, nargs="*", default=[7, 8, 9, 10, 11, 12], help="SF list to include")
    args = parser.parse_args()

    header = (
        "SF",
        "Best Δ(dB)",
        "At accuracy",
        "CS SNR(dB)",
        "LT SNR(dB)",
    )
    print(f"{header[0]:>3}  {header[1]:>11}  {header[2]:>12}  {header[3]:>11}  {header[4]:>11}")
    print("-" * 60)

    for sf in args.sfs:
        data = load_sf_data(sf, args.base_dir)
        if data is None:
            print(f"SF{sf:>2}: data not found under {args.base_dir}/SF{sf}/sf{sf}.json")
            continue
        if "ChirpSmoother" not in data or "LoRa Trimmer" not in data:
            print(f"SF{sf:>2}: ChirpSmoother or LoRa Trimmer data missing")
            continue
        snr_values: List[float] = data["snr_range"]
        cs_rates: List[float] = data["ChirpSmoother"]
        lt_rates: List[float] = data["LoRa Trimmer"]

        best_delta, acc, cs_snr, lt_snr = compute_max_gain(snr_values, cs_rates, lt_rates)
        print(f"{sf:>3}  {best_delta:11.3f}  {acc:12.3f}  {cs_snr:11.3f}  {lt_snr:11.3f}")


if __name__ == "__main__":
    main()
