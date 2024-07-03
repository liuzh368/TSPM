import torch
import numpy as np
from torch import nn, Tensor

class GMSRModel(nn.Module):
    def __init__(self,k,h,input_size):
        super(GMSRModel, self).__init__()

        self.hidden_size = h
        self.pre_v = k

        self.W = nn.Parameter(torch.randn(k*h,k*h), requires_grad=True)
        self.B = nn.Parameter(torch.randn(1,k*h), requires_grad=True)
        self.R = nn.Parameter(torch.randn(1,k*h//2), requires_grad=True)

        self.FC_W=nn.Parameter(torch.randn(input_size,h), requires_grad=True)
        self.FC_B=nn.Parameter(torch.randn(1,h), requires_grad=True)
        self.FC_W2=nn.Parameter(torch.randn(input_size,h), requires_grad=True)
        self.FC_B2=nn.Parameter(torch.randn(1,h), requires_grad=True)


    #def step(self,preH,hiden_states):
    def forward(self, inputs: Tensor, hidden_states): # hidden states  k x batch x hidden_size
        # seq x batch x input_size     input_size x h
        h_return=[]
        seq_len=inputs.shape[0]
        batch=inputs.shape[1]
        hidden_size=inputs.shape[2]

        inputs_emb=torch.matmul(inputs,self.FC_W)+self.FC_B
        inputs_emb=torch.nn.functional.relu(inputs_emb)
        inputs_emb=torch.matmul(inputs_emb,self.FC_W2)+self.FC_B2
        #inputs_emb=torch.tanh(inputs_emb)

        preH = torch.concat([hidden_states[i] for i in range(self.pre_v)],dim=-1)  # batch x (3Xhidden_size)
        preH = torch.tanh(preH)
        for i in range(seq_len):
            cosR=torch.cos(self.R).reshape(1,self.pre_v,-1)
            sinR=torch.sin(self.R).reshape(1,self.pre_v,-1)
            cos_sin_R=torch.concat((cosR,sinR),dim=-1).reshape(1,-1)
            preH = preH*cos_sin_R
            preH_attention=torch.matmul(preH,self.W)+self.B
            preH = torch.reshape(preH,(batch,self.pre_v,-1))  # 分成 三份
            preH_attention=torch.reshape(preH_attention,(batch,self.pre_v,-1))
            attention=torch.nn.functional.softmax(torch.abs(preH_attention/2.),dim=1)
            preH_input=(preH*attention).sum(1)

            inputs_input=inputs_emb[i]/8  # 这里不太清楚这个8是怎么来的
            h_output=inputs_input+preH_input
            h_output=torch.tanh(h_output)
            h_return.append(h_output)
            preH=torch.concat((preH[:,1:,:].view(batch,-1),h_output),dim=-1)

        H=torch.concat(h_return[-self.pre_v:],dim=-1).reshape(batch,self.pre_v,-1).swapaxes(0,1)
        h_return=torch.stack(h_return)
        return h_return,H
