import numpy as np
import os


def read_and_display_file_info(savedir="../data"):
    # 定义文件名
    files_to_read = [
        'Pdata_gowalla_day_slot.npy', 'Count_gowalla_day_slot.npy',
        'Pdata_gowalla_timeslot_0.npy', 'Count_gowalla_timeslot_0.npy',
        'Pdata_gowalla_timeslot_1.npy', 'Count_gowalla_timeslot_1.npy',
        'Pdata_gowalla_timeslot_2.npy', 'Count_gowalla_timeslot_2.npy'
    ]

    # 遍历文件名，读取并打印信息
    for file_name in files_to_read:
        file_path = os.path.join(savedir, file_name)
        try:
            data = np.load(file_path, allow_pickle=True)
            print(f"{file_name} loaded successfully. Shape: {data.shape}")
        except FileNotFoundError:
            print(f"{file_name} not found.")
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")


# 调用函数来读取数据并显示信息
read_and_display_file_info()
