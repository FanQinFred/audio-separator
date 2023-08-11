from audio_separator import Separator
import sys
import torch
print(torch.cuda.is_available())
import time

# 记录开始时间
start_time = time.time()
if len(sys.argv) != 4:
    print("Usage: python main.py <input_file_path> <output_file_path> <output_format>")
    sys.exit(1)

input_file_path = sys.argv[1]
output_dir_path = sys.argv[2]
output_format = sys.argv[3]
# 现在你可以使用input_file_path和output_file_path来操作文件
# Initialize the Separator with the audio file and model name

separator = Separator(audio_file_path = input_file_path,
                      output_dir=output_dir_path,
                      output_format=output_format,
                      model_name='UVR_MDXNET_KARA_2',
                      use_cuda=True)


primary_stem_path, secondary_stem_path = separator.separate()

print(f'Primary stem saved at {primary_stem_path}')
print(f'Secondary stem saved at {secondary_stem_path}')
# 记录结束时间
end_time = time.time()

# 计算代码运行时间（单位：秒）
duration = end_time - start_time

# 打印代码运行时间
print("代码运行时间：", duration, "秒")