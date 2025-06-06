# -*- coding: utf-8 -*-
"""
这个脚本用于加载和检查.mat格式的EEG数据文件。
请先确保您已经安装了必要的库，特别是scipy。
您可以通过命令行安装: pip install scipy numpy
"""

import scipy.io
import numpy as np

# --- 用户需要修改的部分 ---
# 请将 'your_file.mat' 替换为您的.mat文件的实际路径
# 例如: 'C:/Users/YourName/Desktop/data.mat' 或 './data.mat'
MAT_FILE_PATH = './data/data_EEG_AI.mat'


# -------------------------

def inspect_mat_file(file_path):
    """
    加载并检查.mat文件的内容，打印出变量名、类型、形状和数据样本。

    Args:
        file_path (str): .mat文件的路径。
    """
    try:
        # 使用scipy.io.loadmat加载.mat文件
        # 这个函数会将.mat文件中的变量加载到一个Python字典中
        mat_data = scipy.io.loadmat(file_path)

        print(f"文件加载成功: {file_path}")
        print("=" * 60)

        # .mat文件加载后是一个字典，key是MATLAB中的变量名
        # 我们过滤掉scipy自动生成的以'__'开头的内部变量
        variable_keys = [key for key in mat_data.keys() if not key.startswith('__')]

        print(f"文件包含的变量 (Keys): {variable_keys}")
        print("-" * 60)

        # 遍历每个变量，打印其详细信息
        if not variable_keys:
            print("文件中未找到有效的变量。")
            return

        for key in variable_keys:
            print(f"--- 正在分析变量: '{key}' ---")
            variable_data = mat_data[key]

            # 打印变量类型
            print(f"  -> 类型: {type(variable_data)}")

            # 如果是Numpy数组，打印其形状（维度）
            if isinstance(variable_data, np.ndarray):
                print(f"  -> 形状 (Shape): {variable_data.shape}")

                # 打印一小部分示例数据，以便了解其内容
                # 根据维度的不同，打印方式略有不同
                if variable_data.ndim == 1:
                    # 一维数组
                    print(f"  -> 示例数据 (前5个元素): \n{variable_data[:5]}")
                elif variable_data.ndim == 2:
                    # 二维数组 (例如: trials x features)
                    print(f"  -> 示例数据 (前5行, 前5列): \n{variable_data[:5, :5]}")
                elif variable_data.ndim == 3:
                    # 三维数组 (例如: trials x channels x timepoints)
                    print(f"  -> 示例数据 (第一个trial, 前5个channel, 前5个timepoint): \n{variable_data[0, :5, :5]}")
                else:
                    # 更高维度
                    print("  -> 数据维度较高, 仅显示展平后的前10个元素:", variable_data.flatten()[:10])
            else:
                # 如果不是Numpy数组（可能是字符串或数字），直接打印
                print(f"  -> 值: {variable_data}")

            print("\n")

    except FileNotFoundError:
        print(f"错误: 文件未找到。请确认路径 '{file_path}' 是否正确，以及文件是否存在。")
    except NotImplementedError:
        print("错误: 无法读取此.mat文件。")
        print("提示: 这可能是因为文件是使用-v7.3标志在MATLAB中保存的。")
        print("      请尝试使用 'h5py' 库来读取，例如: import h5py; f = h5py.File(file_path, 'r')")
    except Exception as e:
        print(f"加载或分析文件时发生未知错误: {e}")


# --- 运行主函数 ---
if __name__ == "__main__":
    inspect_mat_file(MAT_FILE_PATH)
