import time
import numpy as np
from PyLoRa import PyLoRa

def timing_test_all_methods():
    """Test decoding time for all methods across SF 7-12"""
    
    spreading_factors = [7, 8, 9, 10, 11, 12]
    iterations = 100  # Number of iterations per test
    
    # Method mapping to match the table requirements
    method_mapping = {
        'loraphy_fpa': 'LoRaPHY-FPA',
        'loraphy': 'LoRaPHY-CPA', 
        'MFFT': 'MFFT',
        'hfft_decode': 'HFFT',
        'loratrimmer_decode': 'LoRaTrimmer',
        'fft_ideal_decode_decodev2': 'ChirpSmoother'
    }
    
    # Store results for LaTeX table
    results_table = {}
    
    for method_key in method_mapping:
        results_table[method_key] = {}
    
    print("Symbol Decoding Time Comparison Test")
    print("=" * 60)
    
    for sf in spreading_factors:
        print(f"\nTesting SF={sf}...")
        
        # Initialize LoRa instance
        lora = PyLoRa(sf=sf)
        lora.dataE1, lora.dataE2 = lora.gen_constants()
        
        # Prepare test signals
        sig_ideal = lora.ideal_chirp(f0=0)
        sig_real = lora.real_chirp(f0=0)
        
        # Define methods to test with appropriate signal type
        methods_to_test = [
            ('loraphy_fpa', lambda: lora.loraphy_fpa(sig_real)),
            ('loraphy', lambda: lora.loraphy(sig_real)),
            ('MFFT', lambda: lora.MFFT(sig_ideal)),
            ('hfft_decode', lambda: lora.hfft_decode(sig_real)),
            ('loratrimmer_decode', lambda: lora.loratrimmer_decode(sig_real)),
            ('fft_ideal_decode_decodev2', lambda: lora.fft_ideal_decode_decodev2(sig_ideal)),
        ]
        
        for method_name, method_func in methods_to_test:
            print(f"  Testing {method_mapping[method_name]}...")
            
            times = []
            successful_runs = 0
            
            for i in range(iterations):
                try:
                    start_time = time.perf_counter()
                    method_func()
                    end_time = time.perf_counter()
                    
                    execution_time = end_time - start_time
                    times.append(execution_time)
                    successful_runs += 1
                    
                except Exception as e:
                    print(f"    Error in iteration {i+1}: {e}")
                    continue
            
            if times:
                avg_time_ms = np.mean(times) * 1000  # Convert to milliseconds
                results_table[method_name][f'SF={sf}'] = avg_time_ms
                print(f"    Average time: {avg_time_ms:.2f} ms ({successful_runs}/{iterations} successful)")
            else:
                results_table[method_name][f'SF={sf}'] = None
                print(f"    FAILED - no successful runs")
    
    # Generate LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE OUTPUT")
    print("=" * 80)
    
    # Print LaTeX table
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Symbol Decoding Time Comparison (ms)}")
    print("\\label{tab:decoding_time}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{SF=7} & \\textbf{SF=8} & \\textbf{SF=9} & \\textbf{SF=10}")
    print(" & \\textbf{SF=11} & \\textbf{SF=12} \\\\")
    print("\\midrule")
    
    # Print results for each method
    for method_key, display_name in method_mapping.items():
        row = [display_name]
        for sf in spreading_factors:
            sf_key = f'SF={sf}'
            if sf_key in results_table[method_key] and results_table[method_key][sf_key] is not None:
                time_val = results_table[method_key][sf_key]
                if time_val < 1:
                    row.append(f"{time_val:.2f}")
                else:
                    row.append(f"{time_val:.0f}")
            else:
                row.append("FAIL")
        
        # Format row for LaTeX
        if display_name == "ChirpSmoother":
            print(f"\\textbf{{{row[0]}}} & {' & '.join(row[1:])} \\\\")
        else:
            print(f"{row[0]} & {' & '.join(row[1:])} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Also print raw results for verification
    print("\n" + "=" * 80)
    print("RAW RESULTS (ms)")
    print("=" * 80)
    
    print(f"{'Method':<20}", end="")
    for sf in spreading_factors:
        print(f"{'SF='+str(sf):<10}", end="")
    print()
    
    print("-" * 80)
    
    for method_key, display_name in method_mapping.items():
        print(f"{display_name:<20}", end="")
        for sf in spreading_factors:
            sf_key = f'SF={sf}'
            if sf_key in results_table[method_key] and results_table[method_key][sf_key] is not None:
                time_val = results_table[method_key][sf_key]
                print(f"{time_val:<10.2f}", end="")
            else:
                print(f"{'FAIL':<10}", end="")
        print()

if __name__ == "__main__":
    timing_test_all_methods()