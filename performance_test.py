import time
import numpy as np
from PyLoRa import PyLoRa

def test_performance():
    lora = PyLoRa()
    lora.sf = 7
    lora.dataE1, lora.dataE2 = lora.gen_constants()
    print(lora.dataE1)
    print("...")
    print(lora.dataE2)
    # time.sleep(100)
    # Prepare signals
    sig = lora.ideal_chirp(f0=0)
    sig2 = lora.real_chirp(f0=0)
    
    # Number of iterations
    iterations = 200
    
    # Methods to test
    methods = [
        # ("our_ideal_decode_decodev2", lambda: lora.our_ideal_decode_decodev2(sig)),
        ("fft_ideal_decode_decodev2", lambda: lora.fft_ideal_decode_decodev2(sig)),
        ("MFFT", lambda: lora.MFFT(sig)),
        ("hfft_decode", lambda: lora.hfft_decode(sig2)),
        ("loraphy_fpa", lambda: lora.loraphy_fpa(sig2)),
        ("loraphy_cpa", lambda: lora.loraphy(sig2)),
        ("loratrimmer_decode", lambda: lora.loratrimmer_decode(sig2))
    ]
    
    results = {}
    
    print(f"Performance test with {iterations} iterations for each method:")
    print("-" * 60)
    
    for method_name, method_func in methods:
        print(f"Testing {method_name}...")
        
        times = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            try:
                method_func()
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                continue
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            # Progress indicator
            if (i + 1) % 400 == 0:
                print(f"  Completed {i + 1}/{iterations} iterations")
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            results[method_name] = {
                'avg': avg_time,
                'std': std_time,
                'min': min_time,
                'max': max_time,
                'count': len(times)
            }
        else:
            results[method_name] = None
    
    # Display results
    print("\n" + "=" * 80)
    print("PERFORMANCE RESULTS")
    print("=" * 80)
    
    for method_name, result in results.items():
        if result:
            print(f"\n{method_name}:")
            print(f"  Average time: {result['avg']*1000:.4f} ms")
            print(f"  Std deviation: {result['std']*1000:.4f} ms")
            print(f"  Min time: {result['min']*1000:.4f} ms")
            print(f"  Max time: {result['max']*1000:.4f} ms")
            print(f"  Successful runs: {result['count']}/{iterations}")
        else:
            print(f"\n{method_name}: FAILED")
    
    # Sort by average time
    successful_results = {k: v for k, v in results.items() if v is not None}
    sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['avg'])
    
    print("\n" + "-" * 60)
    print("RANKING (fastest to slowest):")
    print("-" * 60)
    
    for i, (method_name, result) in enumerate(sorted_results, 1):
        print(f"{i}. {method_name}: {result['avg']*1000:.4f} ms")

if __name__ == "__main__":
    test_performance()