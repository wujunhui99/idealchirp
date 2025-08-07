from  PyLoRa import lora
lora.sig = lora.read_file("./cs_sf7_bw0_us.cfile")
idx = lora.detect(0)
idx = lora.sync(idx)
print(idx)
lora.limit_save(start=idx,prefix="./datasets/data/real/7")