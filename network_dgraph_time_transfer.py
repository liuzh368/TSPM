import torch
import torch.nn as nn
import time
import numpy as np
from utils import *
import scipy.sparse as sp
import faiss
from setting import Setting
setting = Setting()
setting.parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DyGraphTimeTransfer(nn.Module):
    def __init__(self, input_size, user_count, hidden_size, hidden_dim, time_segments=3):
        super().__init__()
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.hidden_size = 20
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = 20  # 时间嵌入维度

        self.vecs_use = torch.tensor(np.load("WORK/vecs_use.npy", allow_pickle=True),
                                     dtype=torch.float32).to(setting.device)  # 初始向量

        self.time_embeddings = nn.Parameter(torch.randn(time_segments, self.time_embedding_dim),
                                            requires_grad=True)  # 时间嵌入
        nn.init.xavier_uniform_(self.time_embeddings)  # 使用Xavier初始化时间嵌入

        # 定义 W_out, b_out
        self.time_transfer_out_layer1 = nn.Linear(20 * 2,20)
        self.time_transfer_out_layer2 = nn.Linear(20,20)
        # 定义 W_in, b_in
        self.time_transfer_in_layer1 = nn.Linear(20 * 2, 20)
        self.time_transfer_in_layer2 = nn.Linear(20, 20)

    def map_hour_to_segment(self, hour_tensor):
        hours_in_day = hour_tensor % 24
        time_segments = torch.zeros_like(hours_in_day)
        time_segments[(hours_in_day >= 22) | (hours_in_day < 6)] = 0
        time_segments[(hours_in_day >= 6) & (hours_in_day < 14)] = 1
        time_segments[(hours_in_day >= 14) & (hours_in_day < 22)] = 2
        return time_segments

    # x_t_slot, y_t_slot表示一周内的第几个小时
    def forward(self, x, x_t_slot, y, y_t_slot):
        seq_len, user_len = x.size()
        loc_vecs_use = self.vecs_use  # 特征向量的poi表示

        x_view = torch.reshape(x, (-1,))
        y_view = torch.reshape(y, (-1,))
        x_t_slot_view = torch.reshape(x_t_slot, (-1,))
        y_t_slot_view = torch.reshape(y_t_slot, (-1,))

        # 转换时间槽
        x_t_slot_transformed = self.map_hour_to_segment(x_t_slot_view)
        y_t_slot_transformed = self.map_hour_to_segment(y_t_slot_view)


        x_emb = torch.index_select(loc_vecs_use, 0, x_view)
        x_emb = torch.reshape(x_emb, (seq_len, user_len, -1))  #x.view(-1).cpu().numpy()

        y_emb = torch.index_select(loc_vecs_use, 0, y_view)
        y_emb = torch.reshape(y_emb, (seq_len, user_len, -1))

        # 获取对应的时间嵌入向量
        x_time_emb = self.time_embeddings[x_t_slot_transformed]
        y_time_emb = self.time_embeddings[y_t_slot_transformed]

        # 重新调整时间嵌入向量的形状以匹配x_emb和y_emb
        x_time_emb = x_time_emb.view(seq_len, user_len, -1)
        y_time_emb = y_time_emb.view(seq_len, user_len, -1)

        # 拼接POI嵌入和时间嵌入
        x_combined_emb = torch.cat((x_emb, x_time_emb), dim=-1)
        y_combined_emb = torch.cat((y_emb, y_time_emb), dim=-1)

        # 处理拼接后的向量以生成时间转出嵌入
        xi_out = self.time_transfer_out_layer1(x_combined_emb)
        xi_out = torch.relu(xi_out)
        xi_out = self.time_transfer_out_layer2(xi_out)

        # 处理拼接后的向量以生成时间转入嵌入
        xi_in_pos = self.time_transfer_in_layer1(y_combined_emb)
        xi_in_pos = torch.relu(xi_in_pos)
        xi_in_pos = self.time_transfer_in_layer2(xi_in_pos)

        # 生成负样本：随机选择一个非下一个访问的POI
        neg_indices = torch.randint(0, self.input_size, (seq_len * user_len,)).to(x.device)
        neg_indices = neg_indices.view(seq_len, user_len)
        neg_emb = torch.index_select(loc_vecs_use, 0, neg_indices.view(-1)).view(seq_len, user_len, -1)


        target_time_slot = 0
        # 创建一个与 y_t_slot_transformed 形状相同但所有值都为target_time_slot的张量
        constant_time_slot = torch.full_like(y_t_slot_transformed, target_time_slot)

        # 为负样本生成对应的时间转入嵌入，使用与 constant_time_slot 相同的时间槽索引
        neg_time_emb = self.time_embeddings[constant_time_slot]


        # 为负样本生成对应的时间转入嵌入，使用与正样本相同的时间槽索引
        # neg_time_emb = self.time_embeddings[y_t_slot_transformed]


        neg_time_emb = neg_time_emb.view(seq_len, user_len, -1)
        neg_combined_emb = torch.cat((neg_emb, neg_time_emb), dim=-1)

        xi_in_neg = self.time_transfer_in_layer1(neg_combined_emb)
        xi_in_neg = torch.relu(xi_in_neg)
        xi_in_neg = self.time_transfer_in_layer2(xi_in_neg)

        # 计算损失
        pos_dist = torch.norm(xi_out - xi_in_pos, p=2, dim=-1)
        neg_dist = torch.norm(xi_out - xi_in_neg, p=2, dim=-1)

        loss_function = nn.LogSigmoid()
        loss = -1. * loss_function(neg_dist - pos_dist).mean()  # 对应公式(6)

        return loss




