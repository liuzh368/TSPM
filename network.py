import torch
import torch.nn as nn
from enum import Enum
import time
import numpy as np
from utils import *
import scipy.sparse as sp
import math
import faiss
from GMSR import GMSRModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


class Rnn(Enum):
    """ The available RNN units """

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory():
    """ Creates the desired RNN unit. """

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)  # 因为拼接了time embedding
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)

class Flashback(nn.Module):
    """ Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    # def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory, lambda_loc, lambda_user, use_weight,
    #              graph, use_graph_user, use_spatial_graph, interact_graph, pre_model, model_time_transfer_pre):
    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory, lambda_loc, lambda_user,
                 use_weight,
                 graph, use_graph_user, use_spatial_graph, interact_graph, model_pre_combined):
        super().__init__()
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight

        self.GMSR=GMSRModel(k=6,h=hidden_size,input_size=hidden_size)
        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph


        self.model_pre_combined = model_pre_combined
        # self.pre_model = pre_model
        # self.model_time_transfer_pre = model_time_transfer_pre


        # self.pre_model.vec_embedding_layer1.weight.requires_grad=False
        # self.pre_model.vec_embedding_layer1.bias.requires_grad=False
        # self.pre_model.vec_embedding_layer2.weight.requires_grad=False
        # self.pre_model.vec_embedding_layer2.bias.requires_grad=False

        # self.time_encoder = nn.Embedding(24 * 7, hidden_size)  # time embedding
        self.user_encoder = nn.Embedding(
            user_count, hidden_size)  # user embedding
        self.poiembedding=nn.Embedding(input_size,hidden_size)

        #self.rnn = rnn_factory.create(hidden_size)  # 改了这里！！！
        self.fc = nn.Linear(2 * hidden_size, input_size)

        #self.pre_model.prepare_train()

    def Loss_l2(self):
        base_params = dict(self.named_parameters())
        loss_l2=0.
        count=0
        for key, value in base_params.items():
            if 'bias' not in key and 'pre_model' not in key:
                loss_l2+=torch.sum(value**2)
                count+=value.nelement()
        return loss_l2

    def forward(self, x, t, t_slot, s, y, y_t, y_t_slot, h, active_user):
        device = x.device
        self.model_pre_combined.prepare_train()
        seq_len, user_len = x.size()
        # input_tensorlist=torch.tensor(np.array([i for i in range(self.input_size)]),dtype=int).cuda()
        input_tensorlist = torch.tensor(np.array([i for i in range(self.input_size)]), dtype=int).to(device)

        x_embedding_network=self.poiembedding(input_tensorlist).to(device)


        x_emb = self.model_pre_combined.emb_with_neighbor_output(x, t_slot, y, y_t_slot, x_embedding_network)
        x_emb=torch.reshape(x_emb,(seq_len,user_len,-1)).to(device)

        p_u = self.user_encoder(active_user).to(device)  # (1, user_len, hidden_size)
        p_u = p_u.view(user_len, self.hidden_size).to(device)

        # out 包含第一个时间步到最后一个时间步的所有隐藏状态（20个）
        # h 包含最后 6 个时间步的隐藏状态，保持短期依赖
        out,h = self.GMSR(x_emb,h)
        out = out.to(device)
        h = h.to(device)
        #out,h = self.rnn(x_emb, h[0,...])  # (seq_len, user_len, hidden_size)
        out_w = torch.zeros(seq_len, user_len,
                            self.hidden_size, device=x.device)

        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)  # (200, 1)
            for j in range(i + 1):
                dist_t = t[i] - t[j]
                dist_s = torch.norm(s[i] - s[j], dim=-1).to(device)
                a_j = self.f_t(dist_t, user_len).to(device)  # (user_len, )
                b_j = self.f_s(dist_s, user_len).to(device)
                #b_j = self.f_s(s[i,:,1],s[i,:,0],s[j,:,1],s[j,:,0])
                a_j = a_j.unsqueeze(1).to(device)  # (user_len, 1)
                b_j = b_j.unsqueeze(1).to(device)
                w_j = a_j * b_j + 1e-10  # small epsilon to avoid 0 division
                #w_j = w_j * user_loc_similarity[:, j].unsqueeze(1)  # (user_len, 1)


                sum_w += w_j
                out_w[i] += w_j * out[j]  # (user_len, hidden_size)
            out_w[i] /= sum_w

        out_pu = torch.zeros(seq_len, user_len, 2 *
                             self.hidden_size, device=x.device)
        for i in range(seq_len):
            # (user_len, hidden_size * 2)
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1).to(device)

        y_linear = self.fc(out_pu)  # (seq_len, user_len, loc_count)

        return y_linear, h

def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    """ use fixed normal noise as initialization """

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        self.numberk=6
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len*self.numberk):
            hs.append(self.h0)
        # (1, 200, 10)
        return torch.stack(hs, dim=0).view(1*self.numberk, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return torch.stack([self.h0]*self.numberk,dim=0)


class LstmStrategy(H0Strategy):
    """ creates h0 and c0 using the inner strategy """

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return h, c

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return h, c
