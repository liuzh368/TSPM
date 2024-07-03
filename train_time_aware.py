import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import pickle
from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network import create_h0_strategy
import network_dgraph
import network_time_aware
from evaluation import Evaluation
from tqdm import tqdm
from scipy.sparse import coo_matrix
import os

from datetime import datetime
import pytz

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
        spath=os.path.join(setting.workpath,'Pdata_gowalla.npy')
    Userdata=np.load(spath,allow_pickle=True)
    return Userdata

def readCountdata(path=None):
    if not path:
        path=os.path.join(setting.workpath,'Count_gowalla.npy')
    countdata=np.load(path,allow_pickle=True)
    usercount=countdata[0]
    locscount=countdata[1]
    return usercount,locscount



Userdata=readPdata()  # user time time_slots coords locs
usercount,locscount=readCountdata()


poi_loader = PoiDataloader(
    setting.max_users, setting.min_checkins,setting.work_length,Userdata,usercount,locscount,None)  # 0， 5*20+1
# poi_loader.read(setting.dataset_file)
# print('Active POI number: ', poi_loader.locations())  # 18737 106994
# print('Active User number: ', poi_loader.user_count())  # 32510 7768
# print('Total Checkins number: ', poi_loader.checkins_count())  # 1823598

log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))

dataset = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TRAIN)  # 20, 200 or 1024, 0
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

assert setting.batch_size < poi_loader.user_count(
), 'batch size must be lower than the amount of available users'



#model_pre=network_dgraph.DyGraph(poi_loader.locations(),poi_loader.user_count(),20,setting.hidden_dim).cuda()
# model_pre=network_dgraph.DyGraph(poi_loader.locations(),poi_loader.user_count(),20,setting.hidden_dim)

model_pre = network_time_aware.TimeAwareDyGraph(poi_loader.locations(), poi_loader.user_count(), 20, setting.hidden_dim, 3)

#model_pre.load_state_dict(torch.load('ablation/dyn_network_concat_240.pth'))
# model_pre.cuda()
model_pre.to(setting.device)
#  training loop
optimizer = torch.optim.Adam(model_pre.parameters(
), lr=0.002, weight_decay=setting.weight_decay)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer, milestones=[20, 40, 60, 80], gamma=0.2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[20,40], gamma=0.5)

bar = tqdm(total=2400)
bar.set_description('Training')


for e in range(30):  # 100
    dataset.shuffle_users()  # shuffle users before each epoch!
    losses = []
    epoch_start = time.time()
    for i, (x, t, t_slot, s, y, y_t, y_t_slot, reset_h, active_users) in enumerate(dataloader):
        x = x.squeeze().to(setting.device)
        y = y.squeeze().to(setting.device)


        # 转换时间戳序列并确定时间段
        t_timestamps = t.flatten().cpu().numpy()
        y_t_timestamps = y_t.flatten().cpu().numpy()


        def timestamp_to_hour(timestamp):
            dt = datetime.fromtimestamp(timestamp, tz=pytz.utc)
            return dt.hour


        x_hours = [timestamp_to_hour(ts) for ts in t_timestamps]
        y_hours = [timestamp_to_hour(ts) for ts in y_t_timestamps]

        x_time_segments = []
        y_time_segments = []

        for x_hour, y_hour in zip(x_hours, y_hours):
            if 22 <= x_hour or x_hour < 6:
                x_time_segment = 0
            elif 6 <= x_hour < 14:
                x_time_segment = 1
            else:
                x_time_segment = 2
            x_time_segments.append(x_time_segment)

            if 22 <= y_hour or y_hour < 6:
                y_time_segment = 0
            elif 6 <= y_hour < 14:
                y_time_segment = 1
            else:
                y_time_segment = 2
            y_time_segments.append(y_time_segment)

        x_time_segments = torch.tensor(x_time_segments, dtype=torch.long, device=setting.device).reshape(x.shape[0], -1)
        y_time_segments = torch.tensor(y_time_segments, dtype=torch.long, device=setting.device).reshape(y.shape[0], -1)

        optimizer.zero_grad()
        forward_start = time.time()
        loss = model_pre(x, x_time_segments, y, y_time_segments)  # 此时可以满足传入给network_time_aware模型四个参数

        lossx = loss
        lossx.backward()

        print(lossx)

        losses.append(loss.item())
        optimizer.step()

    scheduler.step()
    bar.update(1)
    epoch_end = time.time()
    log_string(log, 'One training need {:.2f}s'.format(
        epoch_end - epoch_start))
    # statistics:
    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        # print(f'Epoch: {e + 1}/{setting.epochs}')
        # print(f'Used learning rate: {scheduler.get_last_lr()[0]}')
        # print(f'Avg Loss: {epoch_loss}')
        log_string(log, f'Epoch: {e + 1}/{setting.epochs}')
        log_string(log, f'Used learning rate: {scheduler.get_last_lr()[0]}')
        log_string(log, f'Avg Loss: {epoch_loss}')
    if (e+1)%30==0:
        losses = []
        with torch.no_grad():
            for i, (x, t, t_slot, s, y, y_t, y_t_slot, reset_h, active_users) in enumerate(dataloader_test):
                # reset hidden states for newly added users
                x = x.squeeze().to(setting.device)
                y = y.squeeze().to(setting.device)

                # 转换时间戳序列并确定时间段
                t_timestamps = t.flatten().cpu().numpy()
                y_t_timestamps = y_t.flatten().cpu().numpy()


                def timestamp_to_hour(timestamp):
                    dt = datetime.fromtimestamp(timestamp, tz=pytz.utc)
                    return dt.hour


                x_hours = [timestamp_to_hour(ts) for ts in t_timestamps]
                y_hours = [timestamp_to_hour(ts) for ts in y_t_timestamps]

                x_time_segments = []
                y_time_segments = []

                for x_hour, y_hour in zip(x_hours, y_hours):
                    if 22 <= x_hour or x_hour < 6:
                        x_time_segment = 0
                    elif 6 <= x_hour < 14:
                        x_time_segment = 1
                    else:
                        x_time_segment = 2
                    x_time_segments.append(x_time_segment)

                    if 22 <= y_hour or y_hour < 6:
                        y_time_segment = 0
                    elif 6 <= y_hour < 14:
                        y_time_segment = 1
                    else:
                        y_time_segment = 2
                    y_time_segments.append(y_time_segment)

                x_time_segments = torch.tensor(x_time_segments, dtype=torch.long, device=setting.device).reshape(
                    x.shape[0], -1)
                y_time_segments = torch.tensor(y_time_segments, dtype=torch.long, device=setting.device).reshape(
                    y.shape[0], -1)



                forward_start = time.time()
                loss = model_pre(x, x_time_segments, y, y_time_segments)
                losses.append(loss.item())
        epoch_loss = np.mean(losses)
        log_string(log, f'test Loss: {epoch_loss}')

    if (e+1)%30==0:
        torch.save(model_pre.state_dict(),f"WORK/time_aware_dyn_network_{e+1}.pth")
        break



bar.close()
