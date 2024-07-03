import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from setting import Setting
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network_coalition import DyGraphCombinedModel  # 确保该类在一个名为network_coalition.py的文件中
from tqdm import tqdm
from scipy.sparse import coo_matrix

'''
Main train script to invoke from commandline.
'''

setting = Setting()
setting.parse()
log = open(setting.log_file, 'w')
# log_string(log, setting)

print(setting)

log_string(log, 'log_file: ' + setting.log_file)
#log_string(log, 'user_file: ' + setting.trans_user_file)
log_string(log, 'loc_temporal_file: ' + setting.trans_loc_file)
#log_string(log, 'loc_spatial_file: ' + setting.trans_loc_spatial_file)
log_string(log, 'interact_file: ' + setting.trans_interact_file)

log_string(log, str(setting.lambda_user))
log_string(log, str(setting.lambda_loc))

log_string(log, 'W in AXW: ' + str(setting.use_weight))
log_string(log, 'GCN in user: ' + str(setting.use_graph_user))
log_string(log, 'spatial graph: ' + str(setting.use_spatial_graph))


def readPdata(spath=None):
    if not spath:
        spath = os.path.join(setting.workpath, 'Pdata_gowalla.npy')
    Userdata = np.load(spath, allow_pickle=True)
    return Userdata

def readCountdata(path=None):
    if not path:
        path = os.path.join(setting.workpath, 'Count_gowalla.npy')
    countdata = np.load(path, allow_pickle=True)
    usercount = countdata[0]
    locscount = countdata[1]
    return usercount, locscount


Userdata = readPdata()  # user time time_slots coords locs
usercount, locscount = readCountdata()

poi_loader = PoiDataloader(
    setting.max_users, setting.min_checkins, setting.work_length, Userdata, usercount, locscount, None)  # 0， 5*20+1

log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))

dataset = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TRAIN)  # 20, 200 or 1024, 0
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

assert setting.batch_size < poi_loader.user_count(), 'batch size must be lower than the amount of available users'

model_combined = DyGraphCombinedModel(poi_loader.locations(),
                                      poi_loader.user_count(),
                                      20,
                                      setting.hidden_dim,
                                      3)

model_combined.to(setting.device)

# training loop
optimizer = torch.optim.Adam(model_combined.parameters(), lr=0.002, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)

bar = tqdm(total=2400)
bar.set_description('Training')

for e in range(60):  # 100
    dataset.shuffle_users()  # shuffle users before each epoch!
    losses = []
    epoch_start = time.time()
    for i, (x, t, t_slot, s, y, y_t, y_t_slot, reset_h, active_users) in enumerate(dataloader):
        x = x.squeeze().to(setting.device)
        y = y.squeeze().to(setting.device)
        t_slot = t_slot.squeeze().to(setting.device)
        y_t_slot = y_t_slot.squeeze().to(setting.device)
        optimizer.zero_grad()
        forward_start = time.time()
        loss = model_combined(x, t_slot, y, y_t_slot)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    scheduler.step()
    bar.update(1)
    epoch_end = time.time()
    log_string(log, 'One training need {:.2f}s'.format(epoch_end - epoch_start))
    # statistics:
    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        log_string(log, f'Epoch: {e + 1}/{setting.epochs}')
        log_string(log, f'Used learning rate: {scheduler.get_last_lr()[0]}')
        log_string(log, f'Avg Loss: {epoch_loss}')
    if (e + 1) % 60 == 0:
        losses = []
        with torch.no_grad():
            for i, (x, t, t_slot, s, y, y_t, y_t_slot, reset_h, active_users) in enumerate(dataloader_test):
                x = x.squeeze().to(setting.device)
                y = y.squeeze().to(setting.device)
                t_slot = t_slot.squeeze().to(setting.device)
                y_t_slot = y_t_slot.squeeze().to(setting.device)

                forward_start = time.time()
                loss = model_combined(x, t_slot, y, y_t_slot)
                losses.append(loss.item())

        epoch_loss = np.mean(losses)
        log_string(log, f'test Loss: {epoch_loss}')

    if (e + 1) % 60 == 0:
        torch.save(model_combined.state_dict(), f"WORK/dyn_network_combined_gowalla_{e + 1}_without_b.pth")
        break

bar.close()
