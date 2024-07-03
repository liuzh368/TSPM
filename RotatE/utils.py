import numpy as np
import torch

import os
from datetime import datetime
import time
import pytz

from datetime import timezone
from tzwhere import tzwhere
from timezonefinder import TimezoneFinder
class Graph_Embedding:
    def __init__(self,checkin_file_path,Pdata_path,count_path):
        self.checkin_path=checkin_file_path
        self.Pdata_path=Pdata_path
        self.count_path=count_path


    def create_entity_file(self):
        self.user2id={}
        self.poi2id={}
        self.Pdata=np.load(self.Pdata_path,allow_pickle=True)
        self.Countdata=np.load(self.count_path,allow_pickle=True)
        for line in self.Pdata:
            for loc in line[4]:
                if loc not in self.poi2id:
                    self.poi2id[loc]=loc
        poilist=list(self.poi2id.values())
        # np.save("WORK/entity_list_gowalla",np.array(poilist,dtype=np.int32),allow_pickle=True)
        # print("entity is Done")

        # 确保 WORK 目录存在
        if not os.path.exists("../WORK"):
            os.makedirs("../WORK")

        # 现在尝试保存文件
        np.save("../WORK/entity_list_4sq", np.array(poilist, dtype=np.int32), allow_pickle=True)
        print("entity is Done")

    def create_relation_file(self):
        self.relation_dict={"pre_and_sub_and_self":0}
        np.save("../WORK/relation_dict_4sq",self.relation_dict,allow_pickle=True)
        print("relation is Done")

    def create_tuplerelations_file(self):
        # 获取包含时间戳的关系列表
        relation_pre_and_sub = self.precursor_and_subsequent_relations()  # 1,2
        # 分时间段保存关系
        # self.save_relations_by_timeslot(relations_with_timestamp)



        relation_list = relation_pre_and_sub
        relation_array = np.array(relation_list, dtype=np.int32)

        np.save("../WORK/relation_only_pre_and_sub_4sq", relation_array, allow_pickle=True)

    def precursor_and_subsequent_relations(self):
        precursor_dict = {}  # 用于存储先前访问和后续访问的关系
        list_relations = []  # 存储所有关系的列表
        infx = np.load("../data/Pdata_4sq.npy", allow_pickle=True)  # 加载数据，这里假设是所有用户的 位置访问数据
        for userdata in infx:  # 遍历每个用户的数据
            timestamps = userdata[1] # 时间戳列
            # loclen = int(len(userdata[4])*0.8)   # 计算该用户访问位置的80%，用于建立关系
            locs = userdata[4] # 位置列
            for i in range(len(locs)-1):  # 遍历除最后一个位置外的所有位置
                # 对于每对连续访问的POI，记录它们的先后关系
                start = locs[i]  # 当前位置
                end = locs[i+1]  # 下一个位置
                # time_start = timestamps[i]
                # time_end = timestamps[i+1]

                if precursor_dict.get((int(start), int(end))) is None:  # 检查当前位置到下一个位置的关系是否已记录
                    precursor_dict[(int(start), int(end))] = 0  # 未记录则添加到字典中
                if precursor_dict.get((int(end), int(start))) is None:  # 检查下一个位置到当前位置的关系是否已记录
                    precursor_dict[(int(end), int(start))] = 0  # 未记录则添加到字典中

                # # 添加先后关系
                # relations.append([start, self.relation_dict["pre_and_sub_and_self"], end, time_start])
                # # 添加后先关系
                # relations.append([end, self.relation_dict["pre_and_sub_and_self"], end, time_end])

        # return relations

        for i in range(self.Countdata[1]):   # 遍历所有位置
            if precursor_dict.get((i, i)) is None:  # 检查自循环的关系是否已记录
                precursor_dict[(i, i)] = 0  # 未记录则添加到字典中

        for key in precursor_dict:  # 遍历所有记录的关系
            list_relations.append([key[0], self.relation_dict["pre_and_sub_and_self"], key[1]])  # 添加到关系列表中，关系类型使用预设的“pre_and_sub_and_self”
        return list_relations  # 返回所有关系列表

    def save_relations_by_timeslot(self, relations):
        timeslot_relations = {0:[], 1:[], 2:[]}
        for start, relation_type, end, timestamp in relations:
            hour = datetime.utcfromtimestamp(timestamp).hour
            timeslot = self.get_timeslot(hour)
            timeslot_relations[timeslot].append([start, relation_type, end])

        for timeslot, rels in timeslot_relations.items():
            np.save(f"../WORK/relation_only_pre_and_sub_gowalla_new_{timeslot}.npy", np.array(rels, dtype=np.int32), allow_pickle=True)







    def get_timeslot(self, hour):
        if 22 <= hour or hour < 6:
            return 0
        elif 6 <= hour < 14:
            return 1
        else:
            return 2




g_e=Graph_Embedding("../data/checkins-4sq.txt","../data/Pdata_4sq.npy","../data/Count_4sq.npy")
g_e.create_entity_file()
g_e.create_relation_file()
g_e.create_tuplerelations_file()
