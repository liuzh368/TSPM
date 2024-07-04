import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from utils import *
from setting import Setting
import time

setting = Setting()
setting.parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DyGraphCombinedModel(nn.Module):
    def __init__(self, input_size, user_count, hidden_size, hidden_dim, time_segments=4):
        super(DyGraphCombinedModel, self).__init__()
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = 20  # 时间嵌入维度

        # 初始化序列模块
        self.vecs_use = torch.tensor(np.load("WORK/vecs_use_gowalla.npy", allow_pickle=True),
                                     dtype=torch.float32).to(setting.device)  # 初始向量
        self.list_centroids = list(np.load("WORK/list_centroids_gowalla.npy", allow_pickle=True))  # 每个聚类中含有的元素(存的不是序号，是array)
        num_centroids = len(self.list_centroids)
        vecs_emblength = self.vecs_use.shape[1]
        self.I_array = np.load("WORK/I_gowalla.npy", allow_pickle=True)  # 每个向量对应的聚类
        self.list_number = np.load("WORK/list_number_gowalla.npy", allow_pickle=True)
        self.index_list = []

        for i in range(num_centroids):
            index_i = faiss.IndexFlatL2(vecs_emblength)
            index_i.add(self.list_centroids[i])
            self.index_list.append(index_i)

        self.seq_embedding_layer1 = nn.Linear(20 * 5, 40)
        self.seq_embedding_layer2 = nn.Linear(40, 20)

        # 初始化时间增强模块
        self.time_embeddings = nn.Parameter(torch.randn(time_segments, self.time_embedding_dim),
                                            requires_grad=True)  # 时间嵌入
        nn.init.xavier_uniform_(self.time_embeddings)  # 使用Xavier初始化时间嵌入

        self.time_transfer_out_layer1 = nn.Linear(20 * 2, 40)
        self.time_transfer_out_layer2 = nn.Linear(40, 20)
        self.time_transfer_in_layer1 = nn.Linear(20 * 2, 40)
        self.time_transfer_in_layer2 = nn.Linear(40, 20)

    def prepare_train(self):
        #input_tensorlist=torch.tensor(np.array([i for i in range(self.input_size)]),dtype=int).cuda()
        # x_embedding_dy=self.vec_embedding_layer1(self.vecs_use)
        # x_embedding_dy=F.relu(x_embedding_dy)
        # self.x_embedding_dy=self.vec_embedding_layer2(x_embedding_dy)
        self.x_embedding_dy = self.vecs_use

    def map_hour_to_segment(self, hour_tensor):
        hours_in_day = hour_tensor % 24
        time_segments = torch.zeros_like(hours_in_day)
        time_segments[(hours_in_day >= 22) | (hours_in_day < 6)] = 0
        time_segments[(hours_in_day >= 6) & (hours_in_day < 14)] = 1
        time_segments[(hours_in_day >= 14) & (hours_in_day < 22)] = 2
        return time_segments

    def emb_with_neighbor_output(self, x, x_t_slot, y, y_t_slot, x_embedding_network):
        seq_len, user_len = x.size()

        loc_vecs_use = self.vecs_use.to(setting.device)  # 特征向量的poi表示

        x_view = torch.reshape(x, (-1,)).to(setting.device)
        y_view = torch.reshape(y, (-1,)).to(setting.device)
        x_t_slot_view = torch.reshape(x_t_slot, (-1,)).to(setting.device)
        y_t_slot_view = torch.reshape(y_t_slot, (-1,)).to(setting.device)

        # 转换时间槽
        x_t_slot_transformed = self.map_hour_to_segment(x_t_slot_view)

        x_emb = torch.index_select(loc_vecs_use, 0, x_view).to(setting.device)
        x_emb = x_emb.reshape(-1, 20)  # x.view(-1).cpu().numpy()

        # 获取对应的时间嵌入向量
        x_time_emb = self.time_embeddings[x_t_slot_transformed].reshape(-1, 20).to(setting.device)

        # 拼接POI嵌入和时间嵌入
        x_combined_emb = torch.cat((x_emb, x_time_emb), dim=-1).to(setting.device)

        # 处理拼接后的向量以生成时间转出嵌入
        xi_out = self.time_transfer_out_layer1(x_combined_emb).to(setting.device)
        xi_out = torch.relu(xi_out).to(setting.device)
        xi_out = self.time_transfer_out_layer2(xi_out).to(setting.device)

        torch.manual_seed(42)
        target_time_segment = 2
        # 生成与 self.x_embedding_dy 第一维度一致的随机时间槽索引
        rand_time_slot = torch.full((loc_vecs_use.size(0),), target_time_segment, dtype=torch.long).to(setting.device)
        # 获取随机时间槽的时间嵌入
        rand_time_emb = self.time_embeddings[rand_time_slot].view(loc_vecs_use.size(0), -1).to(setting.device)
        # 将所有 POI 的嵌入 self.x_embedding_dy 与时间嵌入进行拼接
        vec_output = torch.cat((loc_vecs_use, rand_time_emb), dim=-1).to(setting.device)
        vec_output = self.time_transfer_in_layer1(vec_output).to(setting.device)
        vec_output = torch.relu(vec_output).to(setting.device)
        vec_output = self.time_transfer_in_layer2(vec_output).to(setting.device)

        # vec_output = torch.cat((loc_vecs_use, rand_time_emb), dim=-1).to(setting.device)
        # vec_output = self.time_transfer_out_layer1(vec_output).to(setting.device)
        # vec_output = torch.relu(vec_output).to(setting.device)
        # vec_output = self.time_transfer_out_layer2(vec_output).to(setting.device)

        x_view = torch.reshape(x, (-1,))
        x_emb = torch.index_select(loc_vecs_use, 0, x_view)
        x_emb = torch.reshape(x_emb, (seq_len, user_len, -1))  # x.view(-1).cpu().numpy()

        x_emb_history = x_emb
        x_emb_history = x_emb_history.reshape(-1, 20).to(setting.device)

        x_emb_history_now = x_emb.reshape(-1, 20).to(setting.device)
        x_emb_history_neg_1 = torch.cat((x_emb[0:1], x_emb[0:seq_len - 1]), dim=0).reshape(-1, 20).to(setting.device)
        x_emb_history_neg_2 = torch.cat((x_emb[0:2], x_emb[0:seq_len - 2]), dim=0).reshape(-1, 20).to(setting.device)
        x_emb_history_neg_3 = torch.cat((x_emb[0:3], x_emb[0:seq_len - 3]), dim=0).reshape(-1, 20).to(setting.device)
        x_emb_history_neg_4 = torch.cat((x_emb[0:4], x_emb[0:seq_len - 4]), dim=0).reshape(-1, 20).to(setting.device)

        x_emb_history_concat = torch.cat(
            (x_emb_history_neg_4, x_emb_history_neg_3, x_emb_history_neg_2, x_emb_history_neg_1, x_emb_history_now),
            dim=-1).to(setting.device)

        x_emb_history_concat = self.seq_embedding_layer1(x_emb_history_concat).to(setting.device)
        x_emb_history_concat = F.relu(x_emb_history_concat).to(setting.device)
        x_emb_history_concat = self.seq_embedding_layer2(x_emb_history_concat).to(setting.device)

        index_x_centroids = self.I_array[x.view(-1).cpu().numpy()].squeeze()  # 获取每个向量所在的聚类

        indexofcentroids = [[] for _ in range(len(self.list_number))]

        for i in range(x_emb_history.shape[0]):
            indexofcentroids[index_x_centroids[i]].append(i)

        return_x_emb = torch.zeros((x_emb_history.shape[0], self.hidden_dim)).to(setting.device)

        for i, line in enumerate(indexofcentroids):
            if len(line) != 0:
                list_nei = self.list_number[i]  # 位于这个聚类内的所有元素
                candidate_number = torch.tensor(list_nei, dtype=int).to(setting.device)
                now_self = x_view[line]

                line_cuda = torch.tensor(np.array(line), dtype=int).to(setting.device)

                # 获取当前行对应的序列嵌入和时间不变性偏好嵌入
                now_self_emb = x_emb_history_concat[line_cuda]  # n x 20
                now_self_time_emb = xi_out[line_cuda]  # n x 20, 时间不变性偏好

                # 获取候选邻居的相应嵌入
                neighbor_time_candidate_emb = torch.index_select(vec_output, 0, candidate_number)  # neigh x 40

                # 扩展维度并计算距离
                now_self_emb = now_self_emb.unsqueeze(1).expand(-1, neighbor_time_candidate_emb.shape[0], -1).to(
                    setting.device)
                now_self_time_emb = now_self_time_emb.unsqueeze(1).expand(-1, neighbor_time_candidate_emb.shape[0],
                                                                          -1).to(setting.device)
                neighbor_time_candidate_emb = neighbor_time_candidate_emb.unsqueeze(0).expand(now_self_emb.shape[0], -1,
                                                                                              -1).to(setting.device)

                distance = torch.norm((now_self_emb + now_self_time_emb) / 2 - neighbor_time_candidate_emb, p=2,
                                      dim=2).to(setting.device)

                rho = 0.02

                score = torch.exp(-rho * distance).to(setting.device)
                values, indices = torch.topk(score, k=min(10, score.shape[-1]), dim=-1, largest=True)
                indices = torch.reshape(indices, (-1,))

                index_embedding_neigh = torch.index_select(candidate_number, 0, indices).to(setting.device)

                embedding_neigh = x_embedding_network[index_embedding_neigh].reshape(values.shape[0], values.shape[1],
                                                                                     x_embedding_network.shape[-1]).to(
                    setting.device)

                self_embedding = x_embedding_network[now_self].unsqueeze(1).to(setting.device)
                embedding_neigh = torch.cat((embedding_neigh, self_embedding), dim=1).to(setting.device)

                one_value = torch.tensor(np.array([1.] * len(line)).reshape(-1, 1), dtype=torch.float32).to(
                    setting.device)
                values = torch.cat((values, one_value), dim=-1).to(setting.device)

                score_new = torch.nn.functional.softmax(values, dim=-1).to(setting.device)
                embedding_return = embedding_neigh * score_new.unsqueeze(-1).to(setting.device)  # 公式(12)后半部分
                embedding_return = embedding_return.sum(1).to(setting.device)
                return_x_emb[line, :] = embedding_return

        return return_x_emb



    def forward(self, x, t_slot, y, y_t_slot):
        seq_len, user_len = x.size()  # 获取输入序列的长度和用户数量

        # 计算序列嵌入
        loc_vecs_use = self.vecs_use  # 获取所有POI的特征向量
        x_view = torch.reshape(x, (-1,)).to(setting.device)  # 将输入x拉伸成一维
        x_emb = torch.index_select(loc_vecs_use, 0, x_view).reshape(seq_len, user_len, -1)  # 获取输入序列中POI的嵌入向量
        x_emb_history_now = x_emb.reshape(-1, 20).to(setting.device)  # 当前时间点的嵌入
        x_emb_history_neg_1 = torch.cat((x_emb[0:1], x_emb[0:seq_len - 1]), dim=0).reshape(-1, 20).to(
            setting.device)  # 前一个时间点的嵌入
        x_emb_history_neg_2 = torch.cat((x_emb[0:2], x_emb[0:seq_len - 2]), dim=0).reshape(-1, 20).to(
            setting.device)  # 前两个时间点的嵌入
        x_emb_history_neg_3 = torch.cat((x_emb[0:3], x_emb[0:seq_len - 3]), dim=0).reshape(-1, 20).to(
            setting.device)  # 前三个时间点的嵌入
        x_emb_history_neg_4 = torch.cat((x_emb[0:4], x_emb[0:seq_len - 4]), dim=0).reshape(-1, 20).to(
            setting.device)  # 前四个时间点的嵌入

        # 将多个时间点的嵌入拼接在一起
        x_emb_history_concat = torch.cat(
            (x_emb_history_neg_4, x_emb_history_neg_3, x_emb_history_neg_2, x_emb_history_neg_1, x_emb_history_now),
            dim=-1).to(setting.device)
        x_emb_history_concat = self.seq_embedding_layer1(x_emb_history_concat).to(setting.device)  # 通过第一个线性层
        x_emb_history_concat = F.relu(x_emb_history_concat).to(setting.device)  # ReLU 激活函数
        x_emb_history_concat = self.seq_embedding_layer2(x_emb_history_concat).to(setting.device)  # 通过第二个线性层

        # 计算正负样本
        index_x_centroids = self.I_array[x_view.cpu().numpy()].squeeze()  # 获取每个POI所属的聚类中心索引
        neg_sample = []  # 负样本列表
        self_sample = []  # 当前样本列表

        indexofcentroids = [[] for _ in range(len(self.list_number))]  # 初始化每个聚类的样本索引列表

        for i in range(x_emb_history_now.shape[0]):
            indexofcentroids[index_x_centroids[i]].append(i)  # 将样本索引添加到对应的聚类中

        for i, line in enumerate(indexofcentroids):
            if len(line) != 0:
                list_nei = self.list_number[i]  # 获取该聚类中的所有POI
                index_len = len(list_nei)
                line_array = np.array(line)
                self_sample.append(line_array)  # 添加当前样本索引

                neg_list = []
                while len(neg_list) < len(line):
                    num_neg = np.random.randint(0, self.input_size, size=(len(line) + 2))
                    index_neg = np.where(self.I_array[num_neg] != i)[0]  # 确保负样本不在同一个聚类中
                    neg_list.extend(num_neg[index_neg])
                neg_list = neg_list[:len(line)]  # 截取前len(line)个负样本
                neg_sample.append(np.array(neg_list))  # 添加负样本索引

        self_sample = torch.tensor(np.concatenate(self_sample), dtype=int).to(setting.device)
        neg_sample = torch.tensor(np.concatenate(neg_sample), dtype=int).to(setting.device)

        self_output = torch.index_select(x_emb_history_concat, 0, self_sample)  # 获取当前样本的嵌入

        # 计算时间增强嵌入
        x_t_slot_view = torch.reshape(t_slot, (-1,)).to(setting.device)
        x_t_slot_transformed = self.map_hour_to_segment(x_t_slot_view)  # 将时间槽转换为时间段
        x_time_emb = self.time_embeddings[x_t_slot_transformed].reshape(-1, 20).to(setting.device)
        x_combined_emb = torch.cat((x_emb_history_now, x_time_emb), dim=-1).to(setting.device)

        xi_out = self.time_transfer_out_layer1(x_combined_emb).to(setting.device)
        xi_out = torch.relu(xi_out).to(setting.device)
        xi_out = self.time_transfer_out_layer2(xi_out).to(setting.device)

        y_view = torch.reshape(y, (-1,)).to(setting.device)
        y_emb = torch.index_select(loc_vecs_use, 0, y_view).reshape(seq_len * user_len, -1)
        y_t_slot_transformed = self.map_hour_to_segment(torch.reshape(y_t_slot, (-1,)).to(setting.device))
        y_time_emb = self.time_embeddings[y_t_slot_transformed].reshape(seq_len * user_len, -1).to(setting.device)
        y_combined_emb = torch.cat((y_emb, y_time_emb), dim=-1).to(setting.device)
        xi_in_pos = self.time_transfer_in_layer1(y_combined_emb).to(setting.device)
        xi_in_pos = torch.relu(xi_in_pos).to(setting.device)
        xi_in_pos = self.time_transfer_in_layer2(xi_in_pos).to(setting.device)
        # xi_in_pos = self.time_transfer_out_layer1(y_combined_emb).to(setting.device)
        # xi_in_pos = torch.relu(xi_in_pos).to(setting.device)
        # xi_in_pos = self.time_transfer_out_layer2(xi_in_pos).to(setting.device)


        neg_view = torch.reshape(neg_sample, (-1,)).to(setting.device)
        neg_emb = torch.index_select(loc_vecs_use, 0, neg_view).reshape(-1, 20)
        neg_time_emb = self.time_embeddings[y_t_slot_transformed[:neg_emb.size(0)]].reshape(-1, 20).to(setting.device)
        neg_combined_emb = torch.cat((neg_emb, neg_time_emb), dim=-1).to(setting.device)
        xi_in_neg = self.time_transfer_in_layer1(neg_combined_emb).to(setting.device)
        xi_in_neg = torch.relu(xi_in_neg).to(setting.device)
        xi_in_neg = self.time_transfer_in_layer2(xi_in_neg).to(setting.device)

        # xi_in_neg = self.time_transfer_out_layer1(neg_combined_emb).to(setting.device)
        # xi_in_neg = torch.relu(xi_in_neg).to(setting.device)
        # xi_in_neg = self.time_transfer_out_layer2(xi_in_neg).to(setting.device)

        # 计算组合损失
        combined_emb = (self_output + xi_out) / 2
        pos_dist = torch.norm(combined_emb - xi_in_pos, p=2, dim=-1)
        neg_dist = torch.norm(combined_emb - xi_in_neg, p=2, dim=-1)
        loss_function = nn.LogSigmoid()
        loss = -loss_function(neg_dist - pos_dist).mean()

        return loss  # 返回损失






