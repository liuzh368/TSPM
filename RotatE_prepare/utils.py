import numpy as np
import torch

import os
from datetime import datetime
import time
import pytz

from datetime import timezone
from tzwhere import tzwhere
from timezonefinder import TimezoneFinder
# import calu_distance


class key_me(object):
    def __init__(self, start, end):
        self.start=start
        self.end=end
    def __hash__(self):
        return hash(str([self.start,self.end]))

    def __eq__(self,other):
        return self.start==other.start and self.end==other.end

class Utils:
    def __init__(self,checkin_file):
        # 我们所需要的唯一文件就是 checkin
        self.checkin_file=checkin_file
        self.settings()
        self.Network_data()
        self.write_poidata()

    def settings(self):
        self.max_users = 0
        self.min_checkins = 101

    def Network_data(self):
        self.user2id = {}
        self.poi2id = {}
        self.poi2gps = {}
        self.users = []
        self.times = []
        self.time_slots = []
        self.coords = []
        self.locs = []
        self.tf = TimezoneFinder(in_memory=True)

    def write_poidata(self):
        self.read_users(self.checkin_file)
        self.read_pois(self.checkin_file)
        self.writedata("../data")
        # self.create_graph()
        # self.loc2loc_array()
        # #self.startiskey_dict_nowaste() may be not useful
        # self.gps_dict()


    def read_users(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')  ## line strip 删除 头尾空格 split 则按照 指定字符分割
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                prev_user = user
                visit_cnt = 1

    def read_pois(self, file):
        f = open(file, 'r')
        lines = f.readlines()

        user_time = []
        user_coord = []
        user_loc = []
        user_time_slot = []

        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)  # from 0
        # tz=tzwhere.tzwhere()
        tf = TimezoneFinder()
        tempcount_candelete=0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue  # user is not of interest(inactive user)
            user = self.user2id.get(user)  # from 0
            tempcount_candelete+=1
            curTime= datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")
            Timestamp=curTime.replace(tzinfo=timezone.utc).timestamp()

            lat = float(tokens[2])  # 纬度
            long = float(tokens[3]) # 经度
            coord = (lat, long)

            # time_zone=tz.tzNameAt(lat,long)

            # 使用 TimezoneFinder 获取时区信息
            time_zone = tf.timezone_at(lng=long, lat=lat)
            if time_zone is None:
                time_zone = self.tf.timezone_at(lng=long, lat=lat)
            time_adjust = datetime.fromtimestamp(Timestamp, pytz.timezone(time_zone))

            time_cyclevalue=time_adjust.weekday()*24+time_adjust.hour # 一周内的具体小时数

            location = int(tokens[4])  # location nr
            if self.poi2id.get(location) is None:  # get-or-set locations
                self.poi2id[location] = len(self.poi2id)
                self.poi2gps[self.poi2id[location]] = coord
            location = self.poi2id.get(location)  # from 0

            if user == prev_user:
                user_time.insert(0, Timestamp)  # insert in front!
                user_time_slot.insert(0, time_cyclevalue)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
            else:
                self.users.append(prev_user)  # 添加用户
                self.times.append(user_time)  # 添加列表
                self.time_slots.append(user_time_slot)
                self.coords.append(user_coord)
                self.locs.append(user_loc)

                prev_user = user
                user_time = [Timestamp]
                user_time_slot = [time_cyclevalue]
                user_coord = [coord]
                user_loc = [location]

        self.users.append(prev_user)
        self.times.append(user_time)
        self.time_slots.append(user_time_slot)
        self.coords.append(user_coord)
        self.locs.append(user_loc)

        assert len(self.users)==len(self.times)==len(self.times)==len(self.time_slots)==len(self.coords)==len(self.locs)
        self.usernum=len(self.user2id)
        self.locnum=len(self.poi2id)
        print(tempcount_candelete)

    def writedata(self,savedir=None):

        # 检查
        print("Starting to write data...")
        if savedir is None:
            savedir = "../data"  # 或者其他默认路径
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            print(f"Created directory: {savedir}")

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        data= np.concatenate((np.array(self.users).reshape(-1,1), np.array(self.times,dtype=object).reshape(-1,1),
                              np.array(self.time_slots,dtype=object).reshape(-1,1), np.array(self.coords,dtype=object).reshape(-1,1), np.array(self.locs,dtype=object).reshape(-1,1)),axis=1)
        np.save(os.path.join(savedir,'Pdata_gowalla'),data,allow_pickle=True)
        self.savepath=os.path.join(savedir,'Pdata')
        self.countdata=np.array([self.usernum,self.locnum],dtype=np.int64)
        np.save(os.path.join(savedir,'Count_gowalla'),self.countdata,allow_pickle=True)

        print(f"Data saved to {savedir}")
        print("Finished writing data.")



class key_me(object):
    def __init__(self, start, end):
        self.start=start
        self.end=end
    def __hash__(self):
        return hash(str([self.start,self.end]))

    def __eq__(self,other):
        return self.start==other.start and self.end==other.end



filepath= "../data/checkins-gowalla.txt"
util=Utils(filepath)  # 预处理
