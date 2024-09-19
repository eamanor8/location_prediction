#! /usr/bin/env python

import os
import datetime
import csv
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data_loader

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
# ==================================================
# ftype = torch.cuda.FloatTensor
# ltype = torch.cuda.LongTensor

# Now use this device in your model and tensors
ftype = torch.FloatTensor().to(device)
ltype = torch.LongTensor().to(device)

# Data loading params
train_file = "./dataset/loc-gowalla_totalCheckins.txt"

# Model Hyperparameters
dim = 13    # dimensionality
ww = 360  # winodw width (6h)
up_time = 1440  # 1d
lw_time = 50    # 50m
up_dist = 100   # ??
lw_dist = 1

# Training Parameters
batch_size = 2
learning_rate = 0.001
momentum = 0.9
evaluate_every = 1

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
user_cnt, poi2id, train_user, train_time, train_lati, train_longi, train_loc, valid_user, valid_time, valid_lati, valid_longi, valid_loc, test_user, test_time, test_lati, test_longi, test_loc = data_loader.load_data(train_file)

#np.save("poi2id_30", poi2id)
print("User/Location: {:d}/{:d}".format(user_cnt, len(poi2id)))
print("==================================================================================")

class STRNNModule(nn.Module):
    def __init__(self):
        super(STRNNModule, self).__init__()

        # embedding:
        self.user_weight = Variable(torch.randn(user_cnt, dim), requires_grad=False).to(device)
        self.h_0 = Variable(torch.randn(dim, 1), requires_grad=False).to(device)
        self.location_weight = nn.Embedding(len(poi2id), dim).to(device)
        self.perm_weight = nn.Embedding(user_cnt, dim).to(device)
        
        # attributes:
        self.time_upper = nn.Parameter(torch.randn(dim, dim).to(device))
        self.time_lower = nn.Parameter(torch.randn(dim, dim).to(device))
        self.dist_upper = nn.Parameter(torch.randn(dim, dim).to(device))
        self.dist_lower = nn.Parameter(torch.randn(dim, dim).to(device))
        self.C = nn.Parameter(torch.randn(dim, dim).to(device))

        # modules:
        self.sigmoid = nn.Sigmoid()

    # find the most closest value to w, w_cap(index)
    def find_w_cap(self, times, i):
        trg_t = times[i] - ww
        tmp_t = times[i]
        tmp_i = i-1
        for idx, t_w in enumerate(reversed(times[:i]), start=1):
            if t_w.data.cpu().numpy() == trg_t.data.cpu().numpy():
                return i-idx
            elif t_w.data.cpu().numpy() > trg_t.data.cpu().numpy():
                tmp_t = t_w
                tmp_i = i-idx
            elif t_w.data.cpu().numpy() < trg_t.data.cpu().numpy():
                if trg_t.data.cpu().numpy() - t_w.data.cpu().numpy() < tmp_t.data.cpu().numpy() - trg_t.data.cpu().numpy():
                    return i-idx
                else:
                    return tmp_i
        return 0

    def return_h_tw(self, times, latis, longis, locs, idx):
        w_cap = self.find_w_cap(times, idx)
        if w_cap == 0:
            return self.h_0
        else:
            self.return_h_tw(times, latis, longis, locs, w_cap)

        lati = latis[idx] - latis[w_cap:idx]
        longi = longis[idx] - longis[w_cap:idx]
        td = times[idx] - times[w_cap:idx]
        ld = self.euclidean_dist(lati, longi)

        data = ','.join(str(e) for e in td.data.cpu().numpy()) + "\t"
        f.write(data)
        data = ','.join(str(e) for e in ld.data.cpu().numpy()) + "\t"
        f.write(data)

        # Check if locs[w_cap:idx] is 0-dimensional
        data = ','.join(str(e.data.cpu().numpy()[0] if e.data.cpu().numpy().ndim > 0 else e.data.cpu().numpy()) for e in locs[w_cap:idx]) + "\t"
        f.write(data)

        # Check if locs[idx] is 0-dimensional
        data = str(locs[idx].data.cpu().numpy()[0] if locs[idx].data.cpu().numpy().ndim > 0 else locs[idx].data.cpu().numpy()) + "\n"
        f.write(data)


    # get transition matrices by linear interpolation
    def get_location_vector(self, td, ld, locs):
        tud = up_time - td
        tdd = td - lw_time
        lud = up_dist - ld
        ldd = ld - lw_dist
        loc_vec = 0
        for i in range(len(tud)):
            Tt = torch.div(torch.mul(self.time_upper, tud[i]) + torch.mul(self.time_lower, tdd[i]), tud[i]+tdd[i])
            Sl = torch.div(torch.mul(self.dist_upper, lud[i]) + torch.mul(self.dist_lower, ldd[i]), lud[i]+ldd[i])
            loc_vec += torch.mm(Sl, torch.mm(Tt, torch.t(self.location_weight(locs[i]))))
        return loc_vec

    def euclidean_dist(self, x, y):
        return torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))

    def forward(self, user, times, latis, longis, locs, step):  # neg_lati, neg_longi, neg_loc, step
        f.write(str(user.data.cpu().numpy()[0]) + "\n")
        # positive sampling
        pos_h = self.return_h_tw(times, latis, longis, locs, len(times)-1)

###############################################################################################
def run(user, time, lati, longi, loc, step):

    user = Variable(torch.from_numpy(np.asarray([user]))).to(device)
    time = Variable(torch.from_numpy(np.asarray(time))).to(device)
    lati = Variable(torch.from_numpy(np.asarray(lati))).to(device)
    longi = Variable(torch.from_numpy(np.asarray(longi))).to(device)
    loc = Variable(torch.from_numpy(np.asarray(loc))).to(device)

    rnn_output = strnn_model(user, time, lati, longi, loc, step)

###############################################################################################
strnn_model = STRNNModule().to(device)

print("Making train file...")
f = open("./prepro_train_%s.txt" % lw_time, 'w')
# Training
train_batches = list(zip(train_time, train_lati, train_longi, train_loc))
for j, train_batch in enumerate(tqdm.tqdm(train_batches, desc="train")):
    batch_time, batch_lati, batch_longi, batch_loc = train_batch
    run(train_user[j], batch_time, batch_lati, batch_longi, batch_loc, step=1)
f.close()

print("Making valid file...")
f = open("./prepro_valid_%s.txt" % lw_time, 'w')
# Evaluating
valid_batches = list(zip(valid_time, valid_lati, valid_longi, valid_loc))
for j, valid_batch in enumerate(tqdm.tqdm(valid_batches, desc="valid")):
    batch_time, batch_lati, batch_longi, batch_loc = valid_batch
    run(valid_user[j], batch_time, batch_lati, batch_longi, batch_loc, step=2)
f.close()

print("Making test file...")
f = open("./prepro_test_%s.txt" % lw_time, 'w')
# Testing
test_batches = list(zip(test_time, test_lati, test_longi, test_loc))
for j, test_batch in enumerate(tqdm.tqdm(test_batches, desc="test")):
    batch_time, batch_lati, batch_longi, batch_loc = test_batch
    run(test_user[j], batch_time, batch_lati, batch_longi, batch_loc, step=3)
f.close()
