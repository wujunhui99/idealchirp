import json


def process_json_file(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 处理数据：除了"snr_range"和"sf"外，所有值都除以4
    for key in data:
        if key not in ["snr_range", "sf"]:
            # 确保值是列表类型
            if isinstance(data[key], list):
                # 将列表中的每个元素除以4
                data[key] = [val / 8 for val in data[key]]

    # 将处理后的数据写入输出文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"处理完成！数据已保存到 {output_file}")


# 主函数
if __name__ == "__main__":
    input_file = "past/sf10.json"
    output_file = "past/nSF10.json"
    process_json_file(input_file, output_file)