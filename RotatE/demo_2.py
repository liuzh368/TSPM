import numpy as np
import torch
from KGEModel import KGEModel  # 确保这个从正确的地方导入

def read_files_and_load_model(savedir, filenames, nentity, nrelation):
    data_contents = {}
    for filename in filenames:
        file_path = f"{savedir}/{filename}"
        if file_path.endswith('.npy'):
            # 特别处理模型文件
            if 'kge_model' in filename:
                model_params = np.load(file_path, allow_pickle=True).item()
                # 创建模型实例
                model = KGEModel(
                    nentity=nentity,
                    nrelation=nrelation,
                    hidden_dim=200,  # 这些参数应与保存模型前一致
                    gamma=6,
                    double_entity_embedding=True,
                    double_relation_embedding=False
                )
                model.load_state_dict(model_params)
                data_contents[filename] = model.state_dict()  # 存储模型参数
            else:
                data = np.load(file_path, allow_pickle=True)
                data_contents[filename] = data
        elif file_path.endswith('.npz'):
            data = np.load(file_path)
            data_contents[filename] = {key: data[key] for key in data.files}
    return data_contents

# Define the directory and filenames to read
savedir = '../WORK'
filenames = [
    'kge_model-gowalla.npy',
    'kge_model-gowalla_0.npy',
    'coo_gowalla_neighbors.npz',
    'coo_gowalla_neighbors_0.npz'
]
nentity = 106994  # This should be the number of entities in your model
nrelation = 1  # This should be the number of relations in your model

# Read and print the contents of the files
file_data = read_files_and_load_model(savedir, filenames, nentity, nrelation)
for filename, content in file_data.items():
    print(f"Contents of {filename}:")
    if isinstance(content, dict):  # Check if the content is from a model or .npz file
        if 'model' in filename:
            print("Model parameters:")  # We print only parameter names and shapes for brevity
            for param_name, param_value in content.items():
                print(f"{param_name}: {param_value.size()}")
        else:
            for key, value in content.items():
                print(f"{key}: shape {value.shape}")
    else:
        print(content)
