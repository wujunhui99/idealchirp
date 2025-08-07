#!/usr/bin/env python3
"""
Test script for FPA (Fine-grained Phase Alignment) implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from PyLoRa import PyLoRa

def test_fpa_vs_cpa():
    """Compare FPA and CPA methods"""
    print("Testing FPA vs CPA methods...")
    
    # Test parameters
    sf = 7
    bw = 125e3
    fs = 1e6
    
    # Create PyLoRa instance
    lora = PyLoRa(sf=sf, bw=bw, fs=fs)
    
    # Test with different symbols
    test_symbols = [0, 1, 15, 31, 63, 127]  # Various symbols for SF7
    
    results = {
        'symbol': [],
        'cpa_detected': [],
        'cpa_value': [],
        'fpa_detected': [],
        'fpa_value': []
    }
    
    for symbol in test_symbols:
        print(f"\nTesting symbol: {symbol}")
        
        # Generate ideal chirp for this symbol
        ideal_sig = lora.ideal_chirp(f0=symbol, iq_invert=0)
        
        # Add some noise to make it more realistic
        snr_db = 10  # 10dB SNR
        noisy_sig = lora.add_noise(ideal_sig, snr_db)
        
        # Test CPA method (existing loraphy)
        cpa_result = lora.loraphy(noisy_sig)
        
        # Test FPA method (new loraphy_fpa)
        fpa_result = lora.loraphy_fpa(noisy_sig)
        
        # Store results
        results['symbol'].append(symbol)
        results['cpa_detected'].append(cpa_result[0])
        results['cpa_value'].append(cpa_result[1])
        results['fpa_detected'].append(fpa_result[0])
        results['fpa_value'].append(fpa_result[1])
        
        print(f"  Original: {symbol}")
        print(f"  CPA detected: {cpa_result[0]}, value: {cpa_result[1]:.2f}")
        print(f"  FPA detected: {fpa_result[0]}, value: {fpa_result[1]:.2f}")
        print(f"  CPA correct: {cpa_result[0] == symbol}")
        print(f"  FPA correct: {fpa_result[0] == symbol}")
    
    # Calculate accuracy
    cpa_accuracy = sum(1 for i, orig in enumerate(results['symbol']) 
                      if results['cpa_detected'][i] == orig) / len(results['symbol'])
    fpa_accuracy = sum(1 for i, orig in enumerate(results['symbol']) 
                      if results['fpa_detected'][i] == orig) / len(results['symbol'])
    
    print(f"\n=== Results Summary ===")
    print(f"CPA Accuracy: {cpa_accuracy*100:.1f}%")
    print(f"FPA Accuracy: {fpa_accuracy*100:.1f}%")
    
    # Calculate average peak values
    avg_cpa_value = np.mean(results['cpa_value'])
    avg_fpa_value = np.mean(results['fpa_value'])
    
    print(f"Average CPA peak value: {avg_cpa_value:.2f}")
    print(f"Average FPA peak value: {avg_fpa_value:.2f}")
    
    return results

def test_phase_misalignment():
    """Test FPA robustness to phase misalignment"""
    print("\n" + "="*50)
    print("Testing phase misalignment robustness...")
    
    sf = 7
    bw = 125e3
    fs = 1e6
    
    lora = PyLoRa(sf=sf, bw=bw, fs=fs)
    
    # Test symbol
    test_symbol = 32
    
    # Generate ideal chirp
    ideal_sig = lora.ideal_chirp(f0=test_symbol, iq_invert=0)
    
    # Test different phase misalignments
    phase_offsets = np.linspace(0, 2*np.pi, 16)
    
    cpa_results = []
    fpa_results = []
    
    for phase_offset in phase_offsets:
        # Create phase misaligned signal
        # Simulate phase misalignment by rotating the second half of the signal
        N = len(ideal_sig)
        misaligned_sig = ideal_sig.copy()
        misaligned_sig[N//2:] *= np.exp(1j * phase_offset)
        
        # Add noise
        noisy_sig = lora.add_noise(misaligned_sig, 10)  # 10dB SNR
        
        # Test both methods
        cpa_result = lora.loraphy(noisy_sig)
        fpa_result = lora.loraphy_fpa(noisy_sig)
        
        cpa_results.append(cpa_result[0] == test_symbol)
        fpa_results.append(fpa_result[0] == test_symbol)
    
    cpa_robustness = np.mean(cpa_results)
    fpa_robustness = np.mean(fpa_results)
    
    print(f"CPA robustness to phase misalignment: {cpa_robustness*100:.1f}%")
    print(f"FPA robustness to phase misalignment: {fpa_robustness*100:.1f}%")
    
    return phase_offsets, cpa_results, fpa_results

if __name__ == "__main__":
    # Run tests
    basic_results = test_fpa_vs_cpa()
    phase_results = test_phase_misalignment()
    
    print("\n" + "="*50)
    print("FPA implementation test completed!")