import os
import numpy as np
from datetime import datetime
import pytz
from collections import defaultdict

class Utils:
    def __init__(self, checkin_file, min_checkins=101, savedir="../data"):
        self.checkin_file = checkin_file
        self.min_checkins = min_checkins
        self.user2id = {}
        self.poi2id = {}
        self.checkins = defaultdict(list)
        self.savedir = savedir

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

    def read_checkins(self):
        with open(self.checkin_file, 'r') as f:
            lines = f.readlines()

        # Process check-ins and accumulate by user
        for line in lines:
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            timestamp_utc = datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)
            lat = float(tokens[2])
            lon = float(tokens[3])
            location = int(tokens[4])

            # Convert user to id
            if user not in self.user2id:
                self.user2id[user] = len(self.user2id)
            user_id = self.user2id[user]

            # Convert poi to id
            if location not in self.poi2id:
                self.poi2id[location] = len(self.poi2id)
            poi_id = self.poi2id[location]

            # Determine the time slot for the check-in
            hour = timestamp_utc.hour
            if 22 <= hour or hour < 6:
                time_slot = 0
            elif 6 <= hour < 14:
                time_slot = 1
            else:
                time_slot = 2

            # Append check-in data
            self.checkins[time_slot].append((user_id, timestamp_utc.timestamp(), hour, lat, lon, poi_id))

    def write_data(self):
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        for time_slot, checkins in self.checkins.items():
            # Filter users based on minimum check-in criteria
            users_checkins = defaultdict(list)
            for checkin in checkins:
                users_checkins[checkin[0]].append(checkin[1:])

            active_users_checkins = {user_id: checkins for user_id, checkins in users_checkins.items() if len(checkins) >= self.min_checkins}

            # Organize data for output
            pdata = []
            for user_id, checkins in active_users_checkins.items():
                timestamps, time_slots, lats, lons, loc_ids = zip(*checkins)
                pdata.append([user_id, list(timestamps), list(time_slots), list(zip(lats, lons)), list(loc_ids)])

            np.save(os.path.join(self.savedir, f'Pdata_gowalla_timeslot_{time_slot}.npy'), np.array(pdata, dtype=object))
            np.save(os.path.join(self.savedir, f'Count_gowalla_timeslot_{time_slot}.npy'), np.array([len(active_users_checkins), len(self.poi2id)], dtype=np.int32))

    def process(self):
        self.read_checkins()
        self.write_data()

# Replace with the path to your check-in data file
checkin_data_file = "../data/checkins-gowalla.txt"

# Initialize Utils class and process the check-in data
utils = Utils(checkin_data_file)
utils.process()
