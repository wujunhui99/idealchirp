from PyLoRa import PyLoRa
import os
lora = PyLoRa()
def mock_data(sf=7):
    dir = os.path.join("./datasets","mock",str(sf))
    if not os.path.exists(dir):
        os.makedirs(dir)
    real_dir = os.path.join(dir,"real")
    ideal_dir = os.path.join(dir, "ideal")
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
    if not os.path.exists(ideal_dir):
        os.makedirs(ideal_dir)
    lora.sf = sf
    for i in range(2 ** sf):
        sig_ideal = lora.ideal_chirp(f0 = i)
        sig_real = lora.real_chirp(f0=i)
        lora.write_file(sig = sig_ideal,file_path=os.path.join(ideal_dir,str(i)+".cfile"))
        lora.write_file(sig=sig_real, file_path=os.path.join(real_dir, str(i) + ".cfile"))

mock_data(sf = 9)
mock_data(sf = 10)
