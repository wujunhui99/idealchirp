from PyLoRa import PyLoRa
import os
lora = PyLoRa()
def mock_data(sf=8):
    dir = os.path.join("./datasets","mock",str(sf))
    if not os.path.exists(dir):
        os.makedirs(dir)
    traditon_dir = os.path.join(dir,"tradition")
    our_dir = os.path.join(dir, "our")
    if not os.path.exists(traditon_dir):
        os.makedirs(traditon_dir)
    if not os.path.exists(our_dir):
        os.makedirs(our_dir)
    lora.sf = sf
    for i in range(2 ** sf):
        sig_our = lora.ideal_chirp(f0 = i)
        sig_tradition = lora.real_chirp(f0=i)
        lora.write_file(sig = sig_our,file_path=os.path.join(our_dir,str(i)+".cfile"))
        lora.write_file(sig=sig_tradition, file_path=os.path.join(traditon_dir, str(i) + ".cfile"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate mock LoRa dataset for given SF')
    parser.add_argument('--sf', type=int, default=10, help='Spreading Factor (default: 10)')
    args = parser.parse_args()
    mock_data(sf=args.sf)
