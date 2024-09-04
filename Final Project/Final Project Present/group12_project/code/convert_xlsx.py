import pandas as pd
import os


def convert_xls_to_xlsx(src_folder, dest_folder):

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        # 检查文件是否为.xls文件
        if filename.endswith('.xls'):
            # 生成完整的文件路径
            file_path = os.path.join(src_folder, filename)
            # 读取.xls文件
            df = pd.read_excel(file_path)
            # 创建新的文件名，替换扩展名为.xlsx
            new_filename = filename.replace('.xls', '.xlsx')
            new_file_path = os.path.join(dest_folder, new_filename)
            # 保存为.xlsx文件
            df.to_excel(new_file_path, index=False)
            print(f"Converted {file_path} to {new_file_path}")


# 指定源文件夹和目标文件夹
src_folder = 'Shanghai_T2DM'
dest_folder = 'Shanghai_T2DM_xlsx'

# 调用函数进行转换
convert_xls_to_xlsx(src_folder, dest_folder)
