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


class DyGraph(nn.Module):
    def __init__(self, input_size, user_count, hidden_size,hidden_dim):
        super().__init__()
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.hidden_size = 20
        self.hidden_dim=hidden_dim
        self.list_centroids=list(np.load("WORK/list_centroids.npy",allow_pickle=True))  #每个聚类中含有的元素(存的不是序号，是array)
        num_centroids=len(self.list_centroids)
        # self.vecs_use=torch.tensor(np.load("WORK/vecs_use.npy",allow_pickle=True),dtype=torch.float32).cuda()  #初始向量
        self.vecs_use = torch.tensor(np.load("WORK/vecs_use.npy", allow_pickle=True),
                                     dtype=torch.float32).to(setting.device)  # 初始向量
        vecs_emblength=self.vecs_use.shape[1]
        self.I_array=np.load("WORK/I.npy",allow_pickle=True)  # 每个向量对应的聚类
        self.list_number=np.load("WORK/list_number.npy",allow_pickle=True)
        self.index_list=[]

        for i in range(num_centroids):
            index_i=faiss.IndexFlatL2(vecs_emblength)
            index_i.add(self.list_centroids[i])
            self.index_list.append(index_i)

        self.seq_embedding_layer1=nn.Linear(20*5,20)
        self.seq_embedding_layer2=nn.Linear(20,20)

        # self.vec_embedding_layer1=nn.Linear(20,20)
        # self.vec_embedding_layer2=nn.Linear(20,20)


    def Loss_l2(self):
        base_params = dict(self.named_parameters())
        loss_l2=0.
        count=0
        for key, value in base_params.items():
            if 'bias' not in key and 'pre_model' not in key:
                loss_l2+=torch.sum(value**2)
                count+=value.nelement()
        return loss_l2

    def prepare_train(self):
        #input_tensorlist=torch.tensor(np.array([i for i in range(self.input_size)]),dtype=int).cuda()
        # x_embedding_dy=self.vec_embedding_layer1(self.vecs_use)
        # x_embedding_dy=F.relu(x_embedding_dy)
        # self.x_embedding_dy=self.vec_embedding_layer2(x_embedding_dy)
        self.x_embedding_dy = self.vecs_use


    def emb_with_neighbor_output(self,x, x_t_slot, y, y_t_slot, x_embedding_network, model_time_transfer_pre):

        seq_len, user_len = x.size()

        loc_vecs_use = self.vecs_use.to(setting.device)  # 特征向量的poi表示

        x_view = torch.reshape(x, (-1,)).to(setting.device)
        y_view = torch.reshape(y, (-1,)).to(setting.device)
        x_t_slot_view = torch.reshape(x_t_slot, (-1,)).to(setting.device)
        y_t_slot_view = torch.reshape(y_t_slot, (-1,)).to(setting.device)

        # 转换时间槽
        x_t_slot_transformed = model_time_transfer_pre.map_hour_to_segment(x_t_slot_view)

        x_emb = torch.index_select(loc_vecs_use, 0, x_view).to(setting.device)
        x_emb = x_emb.reshape(-1, 20)  # x.view(-1).cpu().numpy()

        # 获取对应的时间嵌入向量
        x_time_emb = model_time_transfer_pre.time_embeddings[x_t_slot_transformed].reshape(-1, 20).to(setting.device)

        # 拼接POI嵌入和时间嵌入
        x_combined_emb = torch.cat((x_emb, x_time_emb), dim=-1).to(setting.device)

        # 处理拼接后的向量以生成时间转出嵌入
        xi_out = model_time_transfer_pre.time_transfer_out_layer1(x_combined_emb).to(setting.device)
        xi_out = torch.relu(xi_out).to(setting.device)
        xi_out = model_time_transfer_pre.time_transfer_out_layer2(xi_out).to(setting.device)

        torch.manual_seed(42)
        target_time_segment = 0
        # 生成与 self.x_embedding_dy 第一维度一致的随机时间槽索引
        rand_time_slot = torch.full((loc_vecs_use.size(0),), target_time_segment, dtype=torch.long).to(setting.device)
        # rand_time_slot = torch.randint(0, (loc_vecs_use.size(0),)).to(setting.device)
        # rand_time_slot[x_view] = x_t_slot_transformed # 改变部分时间段与x_t_slot_transformed一致
        # 获取随机时间槽的时间嵌入
        rand_time_emb = model_time_transfer_pre.time_embeddings[rand_time_slot].view(loc_vecs_use.size(0), -1).to(setting.device)
        # 将所有 POI 的嵌入 self.x_embedding_dy 与时间嵌入进行拼接
        vec_output = torch.cat((loc_vecs_use, rand_time_emb), dim=-1).to(setting.device)
        vec_output = model_time_transfer_pre.time_transfer_in_layer1(vec_output).to(setting.device)
        vec_output = torch.relu(vec_output).to(setting.device)
        vec_output = model_time_transfer_pre.time_transfer_in_layer2(vec_output).to(setting.device)


        x_view=torch.reshape(x,(-1,))
        x_emb=torch.index_select(loc_vecs_use,0,x_view)
        x_emb=torch.reshape(x_emb,(seq_len,user_len,-1))  #x.view(-1).cpu().numpy()

        # x_emb_history=x_emb
        x_emb_history=x_emb
        x_emb_history=x_emb_history.reshape(-1,20).to(setting.device)

        x_emb_history_now=x_emb.reshape(-1,20).to(setting.device)
        x_emb_history_neg_1=torch.cat((x_emb[0:1],x_emb[0:seq_len-1]),dim=0).reshape(-1,20).to(setting.device)
        x_emb_history_neg_2=torch.cat((x_emb[0:2],x_emb[0:seq_len-2]),dim=0).reshape(-1,20).to(setting.device)
        x_emb_history_neg_3=torch.cat((x_emb[0:3],x_emb[0:seq_len-3]),dim=0).reshape(-1,20).to(setting.device)
        x_emb_history_neg_4=torch.cat((x_emb[0:4],x_emb[0:seq_len-4]),dim=0).reshape(-1,20).to(setting.device)

        x_emb_history_concat=torch.cat((x_emb_history_neg_4,x_emb_history_neg_3,x_emb_history_neg_2,x_emb_history_neg_1,x_emb_history_now),dim=-1).to(setting.device)

        x_emb_history_concat=self.seq_embedding_layer1(x_emb_history_concat).to(setting.device)
        x_emb_history_concat=F.relu(x_emb_history_concat).to(setting.device)
        x_emb_history_concat=self.seq_embedding_layer2(x_emb_history_concat).to(setting.device)

        index_x_centroids=self.I_array[x.view(-1).cpu().numpy()].squeeze()  #获取每个向量所在的聚类

        # 要注意的是 x_emb_history 是 加权均值
        # 而 index_x_centroids 则是 每个元素的 聚类编号
        # index_list 中存放的 是 用于找topk nei 的 faiss类型参数 其个数等于 聚类个数
        # list_centroids 是一个存放多个 array的 列表 其中存放的就是
        # I_array 存放的也是每个元素所在的聚类编号
        # self.vecs_use 指的是所有的 向量初始值

        list_return=[]
        indexofcentroids=[[]for i in range(len(self.list_number))]

        time_start=time.time()

        # x_emb_history=self.vec_embedding_layer1(x_emb_history).to(setting.device)
        # x_emb_history=F.relu(x_emb_history).to(setting.device)
        # x_emb_history=self.vec_embedding_layer2(x_emb_history).to(setting.device)

        for i in range(x_emb_history.shape[0]):
            indexofcentroids[index_x_centroids[i]].append(i)

        # return_x_emb=torch.zeros((x_emb_history.shape[0],self.hidden_dim)).cuda()
        return_x_emb = torch.zeros((x_emb_history.shape[0], self.hidden_dim)).to(setting.device)

        for i,line in enumerate(indexofcentroids):
            # 如果聚类没有元素 直接跳过
            if len(line)!=0:
                list_nei=self.list_number[i] # 位于这个聚类内的所有元素
                # candidate_number=torch.tensor(list_nei,dtype=int).cuda()
                candidate_number = torch.tensor(list_nei, dtype=int).to(setting.device)
                now_self=x_view[line]

                # line_cuda=torch.tensor(np.array(line),dtype=int).cuda()
                line_cuda = torch.tensor(np.array(line), dtype=int).to(setting.device)

                # 获取当前行对应的序列嵌入和时间不变性偏好嵌入
                now_self_emb = x_emb_history_concat[line_cuda] # n x 10
                now_self_time_emb = xi_out[line_cuda] # n x 10, 时间不变性偏好

                # 获取候选邻居的相应嵌入
                neighbor_candidate_emb = torch.index_select(self.x_embedding_dy,0,candidate_number)  #neigh x 10
                neighbor_time_candidate_emb = torch.index_select(vec_output, 0, candidate_number)


                now_self_emb = now_self_emb.unsqueeze(1).expand(-1, neighbor_candidate_emb.shape[0], -1).to(setting.device)
                neighbor_candidate_emb = neighbor_candidate_emb.unsqueeze(0).expand(now_self_emb.shape[0], -1, -1).to(setting.device)
                distance_e = torch.norm(now_self_emb - neighbor_candidate_emb, p=2, dim=2).to(setting.device)

                now_self_time_emb = now_self_time_emb.unsqueeze(1).expand(-1, neighbor_time_candidate_emb.shape[0], -1).to(setting.device)
                neighbor_time_candidate_emb = neighbor_time_candidate_emb.unsqueeze(0).expand(now_self_emb.shape[0], -1, -1).to(setting.device)
                distance_t = torch.norm(now_self_time_emb - neighbor_time_candidate_emb, p=2, dim=2).to(setting.device)

                rho_1 = 0.015
                rho_2 = 0.005


                score=torch.exp(-rho_1 * distance_e - rho_2 * distance_t).to(setting.device) # 对应公式(7)/(12)   ???  可能需要对这里进行修改？
                values,indices=torch.topk(score,k=min(10,score.shape[-1]),dim=-1,largest=True)
                indices=torch.reshape(indices,(-1,))

                index_embedding_neigh=torch.index_select(candidate_number,0,indices).to(setting.device)

                embedding_neigh=x_embedding_network[index_embedding_neigh].reshape(values.shape[0],values.shape[1],x_embedding_network.shape[-1]).to(setting.device)

                self_embedding=x_embedding_network[now_self].unsqueeze(1).to(setting.device)
                embedding_neigh=torch.concat((embedding_neigh,self_embedding),dim=1).to(setting.device)
                # values=torch.zeros(values.shape).cuda()
                # one_value=torch.tensor(np.array([1.]*len(line)).reshape(-1,1),dtype=torch.float32).cuda()
                one_value = torch.tensor(np.array([1.] * len(line)).reshape(-1, 1), dtype=torch.float32).to(setting.device)
                values=torch.concat((values,one_value),dim=-1).to(setting.device)

                score_new=torch.nn.functional.softmax(values,dim=-1).to(setting.device)
                embedding_return=embedding_neigh*score_new.unsqueeze(-1).to(setting.device) # 公式(12)后半部分
                embedding_return=embedding_return.sum(1).to(setting.device)
                return_x_emb[line,:]=embedding_return
                #print(1)
        time_end=time.time()
        #print((time_end-time_start))
        #return_x_emb=(x_embedding_network[x_view]+return_x_emb*1.0)/2.0
        return return_x_emb

    def forward(self, x, y):
        seq_len, user_len = x.size()
        loc_vecs_use = self.vecs_use  # 特征向量的poi表示

        x_vecs = []
        x_view = torch.reshape(x,(-1,))
        y_view=torch.reshape(y,(-1,))

        x_emb=torch.index_select(loc_vecs_use,0,x_view)
        x_emb=torch.reshape(x_emb,(seq_len,user_len,-1))  #x.view(-1).cpu().numpy()

        y_emb=torch.index_select(loc_vecs_use,0,y_view)
        y_emb=torch.reshape(y_emb,(seq_len,user_len,-1))


        index_x_centroids=self.I_array[x.view(-1).cpu().numpy()].squeeze()  #获取每个向量所在的聚类
        time_start=time.time()


        x_emb_history=x_emb
        x_emb_history=x_emb_history.reshape(-1,20)

        x_emb_history_now=x_emb.reshape(-1,20)
        x_emb_history_neg_1=torch.cat((x_emb[0:1],x_emb[0:seq_len-1]),dim=0).reshape(-1,20)
        x_emb_history_neg_2=torch.cat((x_emb[0:2],x_emb[0:seq_len-2]),dim=0).reshape(-1,20)
        x_emb_history_neg_3=torch.cat((x_emb[0:3],x_emb[0:seq_len-3]),dim=0).reshape(-1,20)
        x_emb_history_neg_4=torch.cat((x_emb[0:4],x_emb[0:seq_len-4]),dim=0).reshape(-1,20)

        x_emb_history_concat=torch.cat((x_emb_history_neg_4,x_emb_history_neg_3,x_emb_history_neg_2,x_emb_history_neg_1,x_emb_history_now),dim=-1)


        # 要注意的是 x_emb_history 是 加权均值
        # 而 index_x_centroids 则是 每个元素的 聚类编号
        # index_list 中存放的 是 用于找topk nei 的 faiss类型参数 其个数等于 聚类个数
        # list_centroids 是一个存放多个 array的 列表 其中存放的就是
        # I_array 存放的也是每个元素所在的聚类编号
        # self.vecs_use 指的是所有的 向量初始值

        pos_sample=[]
        neg_sample=[]
        self_sample=[]

        indexofcentroids=[[]for i in range(len(self.list_number))]

        time_start=time.time()

        x_emb_history_concat=self.seq_embedding_layer1(x_emb_history_concat)
        x_emb_history_concat=F.relu(x_emb_history_concat)
        x_emb_history_concat=self.seq_embedding_layer2(x_emb_history_concat)

        # vec_output=self.vec_embedding_layer1(self.vecs_use)
        # vec_output=F.relu(vec_output)
        # vec_output=self.vec_embedding_layer2(vec_output)

        vec_output = self.vecs_use

        faiss_x_emb_history=x_emb_history.cpu().numpy()

        #self.prepare_train()

        for i in range(x_emb_history.shape[0]):
            indexofcentroids[index_x_centroids[i]].append(i)

        for i,line in enumerate(indexofcentroids):
            if len(line)!=0:
                list_nei=self.list_number[i] # 位于这个聚类内的所有元素
                # candidate_number=torch.tensor(list_nei,dtype=int).cuda()
                # i 既是聚类的编号
                index=self.index_list[i]
                index_len=index.ntotal
                #pos_item=np.random.choice(list_nei,size=(len(line),min(30,index_len)),replace=True)
                # Distance,neighborindex=index.search(faiss_x_emb_history[line],min(index_len,10)) # len(line) x 10
                # neighborindex=neighborindex.reshape(-1)
                #pos_sample.append(self.list_number[i][neighborindex])

                #now_self=x_view[line]
                line_array=np.array(line)
                line_array=np.expand_dims(line_array,axis=1).repeat(min(index_len,10),axis=1)  #扩展多倍
                self_sample.append(line_array.reshape(-1))

                pos_line=np.array(y_view[line].cpu())
                pos_line=np.expand_dims(pos_line,axis=1).repeat(min(index_len,10),axis=1)  #扩展多倍
                pos_sample.append(pos_line.reshape(-1))

                num_neg=np.random.randint(0,self.input_size,size=(min(index_len,10)+2)*len(line))
                index_neg=np.where(self.I_array[num_neg]!=i)[0] # 不在一个聚类里面就是负样本
                now_neg_num=len(index_neg)
                neg_list=[]
                neg_list.extend(num_neg[index_neg])
                while now_neg_num<min(index_len,10)*len(line):
                    num_neg=np.random.randint(0,self.input_size,size=(min(index_len,10)+2)*len(line))
                    index_neg=np.where(self.I_array[num_neg]!=i)[0] # 不在一个聚类里面就是负样本
                    now_neg_num+=len(index_neg)
                    neg_list.extend(num_neg[index_neg])
                neg_list=neg_list[:min(index_len,10)*len(line)]  #这样我们有了10个正样本 10个负样本
                neg_sample.append(np.array(neg_list))


        # pos_sample=torch.tensor(np.concatenate(pos_sample),dtype=int).cuda()
        # neg_sample=torch.tensor(np.concatenate(neg_sample),dtype=int).cuda()
        # self_sample=torch.tensor(np.concatenate(self_sample),dtype=int).cuda()
        pos_sample = torch.tensor(np.concatenate(pos_sample), dtype=int).to(setting.device)
        neg_sample = torch.tensor(np.concatenate(neg_sample), dtype=int).to(setting.device)
        self_sample = torch.tensor(np.concatenate(self_sample), dtype=int).to(setting.device)


        pos_output=torch.index_select(vec_output,0,pos_sample)
        neg_output=torch.index_select(vec_output,0,neg_sample)
        self_output=torch.index_select(x_emb_history_concat,0,self_sample)



        # pos_score=self.score_layer_1(torch.cat((self_output,pos_output),dim=-1)).mean()
        # neg_score=self.score_layer_1(torch.cat((self_output,neg_output),dim=-1)).mean()


        pos_score=torch.norm(pos_output-self_output,p=2,dim=-1)
        neg_score=torch.norm(neg_output-self_output,p=2,dim=-1)

        loss_function=nn.LogSigmoid()
        loss=-1.*loss_function(neg_score-pos_score).mean() # 对应公式(6)
        time_end=time.time()
        #print(time_end-time_start)
        return loss


