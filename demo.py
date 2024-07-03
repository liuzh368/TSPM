import torch

model_path = 'WORK/dyn_network_combined_gowalla_30.pth'

model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

for paramtensor in model_state_dict:
    print(paramtensor, "\t", model_state_dict[paramtensor].size())