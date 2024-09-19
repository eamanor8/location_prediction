#! /usr/bin/env python

import os
import datetime
import math
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
# ==================================================
ftype = torch.FloatTensor if device == torch.device('cpu') else torch.cuda.FloatTensor
ltype = torch.LongTensor if device == torch.device('cpu') else torch.cuda.LongTensor

# Creating a directory to save the trained model's state after training
# Add a directory where the model will be saved
model_save_path = "./saved_model"  # You can change this to any directory you prefer

# Ensure the directory exists
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Data loading params
train_file = "./prepro_train_50.txt"
valid_file = "./prepro_valid_50.txt"
test_file = "./prepro_test_50.txt"

# Model Hyperparameters
dim = 13    # dimensionality
ww = 360  # winodw width (6h)
up_time = 560632.0  # min
lw_time = 0.
up_dist = 457.335   # km
lw_dist = 0.
reg_lambda = 0.1

# Training Parameters
batch_size = 2
num_epochs = 30
learning_rate = 0.001
momentum = 0.9
evaluate_every = 1
h_0 = Variable(torch.randn(dim, 1), requires_grad=False).to(device)

user_cnt = 32899 #50 #107092#0
loc_cnt = 1115406 #50 #1280969#0

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
train_user, train_td, train_ld, train_loc, train_dst = data_loader.treat_prepro(train_file, step=1)
valid_user, valid_td, valid_ld, valid_loc, valid_dst = data_loader.treat_prepro(valid_file, step=2)
test_user, test_td, test_ld, test_loc, test_dst = data_loader.treat_prepro(test_file, step=3)

print("User/Location: {:d}/{:d}".format(user_cnt, loc_cnt))
print("==================================================================================")

class STRNNCell(nn.Module):
    def __init__(self, hidden_size):
        super(STRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # C
        self.weight_th_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # T
        self.weight_th_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # T
        self.weight_sh_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # S
        self.weight_sh_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # S

        self.location_weight = nn.Embedding(loc_cnt, hidden_size)
        self.permanet_weight = nn.Embedding(user_cnt, hidden_size)

        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, td_upper, td_lower, ld_upper, ld_lower, loc, hx):
        loc_len = len(loc)
        Ttd = [((self.weight_th_upper*td_upper[i] + self.weight_th_lower*td_lower[i])\
                /(td_upper[i]+td_lower[i])) for i in range(loc_len)]
        Sld = [((self.weight_sh_upper*ld_upper[i] + self.weight_sh_lower*ld_lower[i])\
                /(ld_upper[i]+ld_lower[i])) for i in range(loc_len)]

        loc = self.location_weight(loc).view(-1,self.hidden_size,1)
        loc_vec = torch.sum(torch.cat([torch.mm(Sld[i], torch.mm(Ttd[i], loc[i]))\
                .view(1,self.hidden_size,1) for i in range(loc_len)], dim=0), dim=0)
        usr_vec = torch.mm(self.weight_ih, hx)
        hx = loc_vec + usr_vec # hidden_size x 1
        return self.sigmoid(hx)

    def loss(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, dst, hx):
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        q_v = self.location_weight(dst)
        output = torch.mm(q_v, (h_tq + torch.t(p_u)))

        return torch.log(1+torch.exp(torch.neg(output)))

    def validation(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, dst, hx):
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        user_vector = h_tq + torch.t(p_u)
        ret = torch.mm(self.location_weight.weight, user_vector).data.cpu().numpy()
        return np.argsort(np.squeeze(-1*ret))

###############################################################################################
def parameters():
    params = []
    for model in [strnn_model]:
        params += list(model.parameters())

    return params

def print_score(batches, step):
    recall1 = 0.
    recall5 = 0.
    recall10 = 0.
    recall100 = 0.
    recall1000 = 0.
    recall10000 = 0.
    iter_cnt = 0

    for batch in tqdm.tqdm(batches, desc="validation"):
        batch_user, batch_td, batch_ld, batch_loc, batch_dst = batch
        if len(batch_loc) < 3:
            continue
        iter_cnt += 1
        batch_o, target = run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=step)

        recall1 += target in batch_o[:1]
        recall5 += target in batch_o[:5]
        recall10 += target in batch_o[:10]
        recall100 += target in batch_o[:100]
        recall1000 += target in batch_o[:1000]
        recall10000 += target in batch_o[:10000]

    print("recall@1: ", recall1/iter_cnt)
    print("recall@5: ", recall5/iter_cnt)
    print("recall@10: ", recall10/iter_cnt)
    print("recall@100: ", recall100/iter_cnt)
    print("recall@1000: ", recall1000/iter_cnt)
    print("recall@10000: ", recall10000/iter_cnt)

###############################################################################################
def run(user, td, ld, loc, dst, step):

    optimizer.zero_grad()

    seqlen = len(td)
    user = Variable(torch.from_numpy(np.asarray([user]))).to(device)

    rnn_output = h_0
    for idx in range(seqlen-1):
        td_upper = Variable(torch.from_numpy(np.asarray(up_time-td[idx]))).to(device)
        td_lower = Variable(torch.from_numpy(np.asarray(td[idx]-lw_time))).to(device)
        ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-ld[idx]))).to(device)
        ld_lower = Variable(torch.from_numpy(np.asarray(ld[idx]-lw_dist))).to(device)
        location = Variable(torch.from_numpy(np.asarray(loc[idx]))).to(device)
        rnn_output = strnn_model(td_upper, td_lower, ld_upper, ld_lower, location, rnn_output)

    td_upper = Variable(torch.from_numpy(np.asarray(up_time-td[-1]))).to(device)
    td_lower = Variable(torch.from_numpy(np.asarray(td[-1]-lw_time))).to(device)
    ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-ld[-1]))).to(device)
    ld_lower = Variable(torch.from_numpy(np.asarray(ld[-1]-lw_dist))).to(device)
    location = Variable(torch.from_numpy(np.asarray(loc[-1]))).to(device)

    if step > 1:
        return strnn_model.validation(user, td_upper, td_lower, ld_upper, ld_lower, location, dst[-1], rnn_output), dst[-1]

    destination = Variable(torch.from_numpy(np.asarray([dst[-1]]))).to(device)
    J = strnn_model.loss(user, td_upper, td_lower, ld_upper, ld_lower, location, destination, rnn_output)

    J.backward()
    optimizer.step()

    return J.data.cpu().numpy()

###############################################################################################
strnn_model = STRNNCell(dim).to(device)
optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=momentum, weight_decay=reg_lambda)

for i in range(num_epochs):
    total_loss = 0.
    train_batches = list(zip(train_user, train_td, train_ld, train_loc, train_dst))
    for j, train_batch in enumerate(tqdm.tqdm(train_batches, desc="train")):
        batch_user, batch_td, batch_ld, batch_loc, batch_dst = train_batch
        if len(batch_loc) < 3:
            continue
        total_loss += run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=1)
    if (i+1) % evaluate_every == 0:
        print("==================================================================================")
        valid_batches = list(zip(valid_user, valid_td, valid_ld, valid_loc, valid_dst))
        print_score(valid_batches, step=2)

print("Training End..")
print("==================================================================================")
print("Test: ")
test_batches = list(zip(test_user, test_td, test_ld, test_loc, test_dst))
print_score(test_batches, step=3)

# After training is complete, save the model's state
model_file_path = os.path.join(model_save_path, 'strnn_model.pth')
torch.save(strnn_model.state_dict(), model_file_path)
print(f"Model saved to {model_file_path}")
