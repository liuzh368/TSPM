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


class TimeAwareDyGraph(nn.Module):
    def __init__(self, input_size, user_count, hidden_size, hidden_dim, time_segments=3):
        super().__init__()

        self.input_size = input_size  # POI个数
        self.user_count = user_count  # 用户数量
        self.hidden_size = 20  # 隐藏层大小
        self.time_embedding_dim = 20  # 时间嵌入维度

        self.list_centroids = list(np.load("WORK/list_centroids.npy", allow_pickle=True))  # 聚类中心列表
        # self.vecs_use = {i: torch.tensor(np.load(f"WORK/vecs_use_{i}.npy", allow_pickle=True), dtype=torch.float32).cuda() for i in range(time_segments)}  # 向量使用
        self.vecs_use = {
            i: torch.tensor(np.load(f"WORK/vecs_use_{i}.npy", allow_pickle=True), dtype=torch.float32).to(setting.device) for i in
            range(time_segments)}  # 向量使用
        self.I_array = np.load("WORK/I.npy", allow_pickle=True)  # 向量对应的聚类
        self.list_number = np.load("WORK/list_number.npy", allow_pickle=True)  # 列表编号
        self.index_lists = [[] for _ in range(time_segments)]  # 索引列表


        for t in range(time_segments):
            num_centroids = len(self.vecs_use[t])
            vecs_emblength = self.vecs_use[t].shape[1]
            for i in range(num_centroids):
                index_i = faiss.IndexFlatL2(vecs_emblength)
                index_i.add(self.vecs_use[t][i].unsqueeze(0).cpu().numpy())  # 注意，需要添加.cpu().numpy()以从Tensor转换为NumPy数组
                self.index_lists[t].append(index_i)

        self.time_embeddings = nn.Parameter(torch.randn(time_segments, self.time_embedding_dim), requires_grad=True)  # 时间嵌入

        self.time_aware_embedding_layer1 = nn.Linear(20 * 3, 20)
        self.time_aware_embedding_layer2 = nn.Linear(20, 10)

        self.vec_embedding_layer1 = nn.Linear(20,20)
        self.vec_embedding_layer2 = nn.Linear(20,10)

        self.positive_sample_adjust_layer = nn.Linear(20, 10)



    def forward(self, x, x_time_segments, y, y_time_segments):
        seq_len, user_len = x.size()

        x_view = x.reshape(-1)
        y_view = y.reshape(-1)
        x_time_segments_view = x_time_segments.reshape(-1)
        y_time_segments_view = y_time_segments.reshape(-1)

        # 初始化全尺寸的嵌入向量，先填充零，这里使用的隐藏维度是20
        x_emb = torch.zeros((x_view.size(0), 20), device=setting.device)
        y_emb = torch.zeros((y_view.size(0), 20), device=setting.device)

        # 遍历所有时间段，填充对应时间段的嵌入向量
        for t in range(len(self.vecs_use)):
            # 获取当前时间段的POI向量
            current_time_vecs = self.vecs_use[t]

            # 处理x的嵌入
            mask_x = (x_time_segments_view == t)
            current_indices_x = x_view[mask_x]
            if current_indices_x.numel() > 0:
                x_emb[mask_x, :] = current_time_vecs[
                    current_indices_x]  # 使用mask来确定每个时间段的具体索引位置，然后只更新这些位置的值。这样可以确保原始的顺序被完整保留。

            # 处理y的嵌入
            mask_y = (y_time_segments_view == t)
            current_indices_y = y_view[mask_y]
            if current_indices_y.numel() > 0:
                y_emb[mask_y, :] = current_time_vecs[
                    current_indices_y]  # 使用mask来确定每个时间段的具体索引位置，然后只更新这些位置的值。这样可以确保原始的顺序被完整保留。

        # 重塑x_emb和y_emb以匹配输入x和y的形状
        x_emb = x_emb.reshape(seq_len, user_len, -1)
        y_emb = y_emb.reshape(seq_len, user_len, -1)

        index_x_centroids = self.I_array[x.view(-1).cpu().numpy()].squeeze()  #获取每个向量所在的聚类

        # 下面我们使用拼接的嵌入向量来构建时间不变性偏好embedding
        xi_list = []  # 用于存储所有时间不变性偏好embedding

        for i in range(seq_len):
            for j in range(user_len):
                # 获取当前时刻及下一时刻的POI和时间嵌入
                current_time_segment = x_time_segments[i, j]
                next_time_segment = y_time_segments[i, j]

                e_i = x_emb[i, j]  # 当前时间点的POI嵌入
                t_i = self.time_embeddings[current_time_segment]  # 当前时间点的时间嵌入
                # e_j = y_emb[i, j]  # 下一个时间点的POI嵌入
                t_j = self.time_embeddings[next_time_segment]  # 下一个时间点的时间嵌入

                # 拼接四者得到时间不变性偏好embedding的输入
                xi_input = torch.cat((e_i, t_i, t_j), dim=0)

                # 通过两层网络计算时间不变性偏好embedding
                xi = self.time_aware_embedding_layer1(xi_input)
                xi = F.relu(xi)
                xi = self.time_aware_embedding_layer2(xi)
                xi_list.append(xi)

        # 把xi_list转换为tensor
        xi_tensor = torch.stack(xi_list, dim=0).view(seq_len, user_len, -1)

        vec_0_output = self.vec_embedding_layer1(self.vecs_use[0])
        vec_0_output = F.relu(vec_0_output)
        vec_0_output = self.vec_embedding_layer2(vec_0_output)

        vec_1_output = self.vec_embedding_layer1(self.vecs_use[1])
        vec_1_output = F.relu(vec_1_output)
        vec_1_output = self.vec_embedding_layer2(vec_1_output)

        vec_2_output = self.vec_embedding_layer1(self.vecs_use[2])
        vec_2_output = F.relu(vec_2_output)
        vec_2_output = self.vec_embedding_layer2(vec_2_output)

        # 对每个样本，根据其时间段选择嵌入源
        negative_samples = []
        for t in range(seq_len):
            for j in range(user_len):
                current_time_segment = x_time_segments[t, j].item()  # 获取当前样本的时间段
                if current_time_segment == 0:
                    vec_output = vec_0_output
                elif current_time_segment == 1:
                    vec_output = vec_1_output
                elif current_time_segment == 2:
                    vec_output = vec_2_output

                # 从对应的嵌入中随机选择负样本S
                negative_sample_indices = torch.randint(0, vec_output.size(0), (10,), device=setting.device)
                negative_samples.append(torch.index_select(vec_output, 0, negative_sample_indices))

        # negative_samples 现在包含根据时间段正确选择的负样本

        # 初始化损失值
        loss = 0

        # 遍历每个样本
        for i in range(seq_len):
            for j in range(user_len):
                current_input_embedding = xi_tensor[i, j]  # 当前样本的时间不变性偏好嵌入

                # positive_sample = self.positive_sample_adjust_layer(y_emb[i, j])  # 正样本是下一个时间点的POI嵌入
                positive_sample = self.vec_embedding_layer1(y_emb[i, j])
                positive_sample = F.relu(positive_sample)
                positive_sample = self.vec_embedding_layer2(positive_sample)



                negative_sample_tensor = negative_samples[i * user_len + j]  # 直接获取负样本tensor

                # 计算与正样本的距离
                pos_distance = torch.norm(current_input_embedding - positive_sample, p=2)

                # 计算与负样本的距离
                neg_distances = torch.norm(current_input_embedding.unsqueeze(0) - negative_sample_tensor, p=2, dim=1)

                # 对数Sigmoid损失
                loss += -torch.sum(torch.log(torch.sigmoid(neg_distances - pos_distance)))

        # 平均损失
        loss /= (seq_len * user_len)

        # 如果有需要添加正则化项
        lambda_reg = 0.01  # 正则化系数
        regularization_term = sum(torch.norm(param, p=2) for param in self.parameters())
        loss += lambda_reg * regularization_term

        return loss



