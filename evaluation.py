import torch
import numpy as np
from utils import log_string

from datetime import datetime
import pytz


class Evaluation:
    """
    Handles evaluation on a given POI dataset and loader.

    The two metrics are MAP and recall@n. Our model predicts sequence of
    next locations determined by the sequence_length at one pass. During evaluation we
    treat each entry of the sequence as single prediction. One such prediction
    is the ranked list of all available locations and we can compute the two metrics.

    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.

    Using the --report_user argument one can access the statistics per user.
    """

    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting, log):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting
        self._log = log

    def map_hour_to_segment(self, hour_tensor):
        hours_in_day = hour_tensor % 24
        time_segments = torch.zeros_like(hours_in_day)
        time_segments[(hours_in_day >= 22) | (hours_in_day < 6)] = 0
        time_segments[(hours_in_day >= 6) & (hours_in_day < 14)] = 1
        time_segments[(hours_in_day >= 14) & (hours_in_day < 22)] = 2

        return time_segments

    def filter_by_time(self, y_t_slot, target_time_segment):
        # 转换时间槽
        y_t_slot_transformed = self.map_hour_to_segment(y_t_slot)

        mask_time = (y_t_slot_transformed == target_time_segment)

        return mask_time


    def evaluate(self):
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)

        with torch.no_grad():
            iter_cnt = 0
            recall1 = 0
            recall5 = 0
            recall10 = 0
            average_precision = 0.

            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)
            reset_count = torch.zeros(self.user_count)

            for i, (x, t, t_slot, s, y, y_t, y_t_slot, reset_h, active_users) in enumerate(self.dataloader):
                active_users = active_users.squeeze()
                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[:, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        reset_count[active_users[j]] += 1

                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze().to(self.setting.device)
                t = t.squeeze().to(self.setting.device)
                t_slot = t_slot.squeeze().to(self.setting.device)
                s = s.squeeze().to(self.setting.device)

                y = y.squeeze().to(self.setting.device)
                y_t = y_t.squeeze().to(self.setting.device)
                y_t_slot = y_t_slot.squeeze().to(self.setting.device)

                active_users = active_users.to(self.setting.device)


                # evaluate:
                out, hp = self.trainer.evaluate(x, t, t_slot, s, y, y_t, y_t_slot, h, active_users)

                for j in range(self.setting.batch_size):
                    # o contains a per user list of votes for all locations for each sequence entry
                    o = out[j]  # (seq_len, loc_count)

                    # partition elements
                    o_n = o.cpu().detach().numpy()  # (seq_len, loc_count)
                    ind = np.argpartition(o_n, -10, axis=1)[:, -10:]  # top 10 elements  # (seq_len, top_10)

                    # 修改部分
                    target_time_segment = 2
                    y_t_slot_mask = self.filter_by_time(y_t_slot, target_time_segment)
                    y_j_time_mask = y_t_slot_mask[:, j]
                    y_j = y[:, j]  # (seq_len, )

                    # 遍历 y_j_time_mask 中为 True 的位置
                    valid_indices = torch.nonzero(y_j_time_mask).squeeze()
                    if valid_indices.dim() == 0:
                        valid_indices = valid_indices.unsqueeze(0)

                    # for k in valid_indices: # 使用这行代码需要分时间段进行评估
                    for k in range(len(y_j)):
                        if reset_count[active_users[j]] > 1:
                            continue  # skip already evaluated users.

                        # resort indices for k:
                        ind_k = ind[k]
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)]  # sort top 10 elements descending

                        r = torch.tensor(r).to(t.device)
                        t = y_j[k]

                        # compute MAP:
                        r_kj = o_n[k, :]
                        t_val = r_kj[t]  # 真实标签y所对应的预测值
                        upper = np.where(r_kj > t_val)[0]
                        precision = 1. / (1 + len(upper))

                        # store
                        u_iter_cnt[active_users[j]] += 1
                        u_recall1[active_users[j]] += t in r[:1]
                        u_recall5[active_users[j]] += t in r[:5]
                        u_recall10[active_users[j]] += t in r[:10]
                        u_average_precision[active_users[j]] += precision

            formatter = "{0:.8f}"
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                recall1 += u_recall1[j]
                recall5 += u_recall5[j]
                recall10 += u_recall10[j]
                average_precision += u_average_precision[j]

                if self.setting.report_user > 0 and (j + 1) % self.setting.report_user == 0:
                    print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1',
                          formatter.format(u_recall1[j] / u_iter_cnt[j]), 'MAP',
                          formatter.format(u_average_precision[j] / u_iter_cnt[j]), sep='\t')

            # print('recall@1:', formatter.format(recall1 / iter_cnt))
            # print('recall@5:', formatter.format(recall5 / iter_cnt))
            # print('recall@10:', formatter.format(recall10 / iter_cnt))
            # print('MAP', formatter.format(average_precision / iter_cnt))
            # print('predictions:', iter_cnt)

            log_string(self._log, 'recall@1: ' + formatter.format(recall1 / iter_cnt))
            log_string(self._log, 'recall@5: ' + formatter.format(recall5 / iter_cnt))
            log_string(self._log, 'recall@10: ' + formatter.format(recall10 / iter_cnt))
            log_string(self._log, 'MAP: ' + formatter.format(average_precision / iter_cnt))
            print('predictions:', iter_cnt)
