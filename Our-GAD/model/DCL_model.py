#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

## copy from DCL-pytorch
import torch
import torch.nn as nn
from model.layers import GraphCNN, AvgReadout, Discriminator, GraphLambdaLayer
import sys
import pickle
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
sys.path.append('.')
# from torch_geometric.utils import k_hop_subgraph

from model.GCL.models import get_sampler, SingleBranchContrast, DualBranchContrast, SupervisedContrast
import model.GCL.losses as L
from utils import laplace_decomp

class DCI(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
        super(DCI, self).__init__()
        self.device = device
        self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(hidden_dim)

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2, cluster_info, cluster_num):
        h_1 = self.gin(seq1, adj)
        h_2 = self.gin(seq2, adj)

        loss = 0
        batch_size = 1
        criterion = nn.BCEWithLogitsLoss()
        for i in range(cluster_num):
            node_idx = cluster_info[i]

            h_1_block = torch.unsqueeze(h_1[node_idx], 0)
            c_block = self.read(h_1_block, msk)
            c_block = self.sigm(c_block)
            h_2_block = torch.unsqueeze(h_2[node_idx], 0)

            lbl_1 = torch.ones(batch_size, len(node_idx))
            lbl_2 = torch.zeros(batch_size, len(node_idx))
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)

            ret = self.disc(c_block, h_1_block, h_2_block, samp_bias1, samp_bias2)
            loss_tmp = criterion(ret, lbl)
            loss += loss_tmp

        return loss / cluster_num

    def get_emb(self, seq1, adj):
        h_1 = self.gin(seq1, adj)
        return h_1


class DGI(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
        super(DGI, self).__init__()
        self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(hidden_dim)

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2, lbl):
        criterion = nn.BCEWithLogitsLoss()
        h_1 = torch.unsqueeze(self.gin(seq1, adj), 0)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = torch.unsqueeze(self.gin(seq2, adj), 0)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        loss = criterion(ret, lbl)

        return loss

class MLP(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MLP, self).__init__()
        self.norm_layer = nn.BatchNorm1d(dataset.data.x.size()[1], affine=False)
        self.final_dropout = args.final_dropout
        self.linear_prediction_1 = nn.Linear(dataset.num_features, args.hidden)
        self.linear_prediction_2 = nn.Linear(args.hidden, 300)
        self.linear_prediction_3 = nn.Linear(300, 64)
        self.linear_prediction = nn.Linear(64, dataset.num_classes)
        self.dropout = args.dropout
        self.softmax = torch.nn.Softmax(dim=1)
        # self.softmax = nn.LogSoftmax(dim=1)

    def m_loss(self):
        return nn.CrossEntropyLoss()

    def forward(self, data):
        x, edge_index = self.norm_layer(data.x), data.edge_index
        x = F.elu(F.dropout(self.linear_prediction_1(x), training=self.training))
        x = F.elu(F.dropout(self.linear_prediction_2(x), training=self.training))
        x = F.elu(F.dropout(self.linear_prediction_3(x), training=self.training))
        x = F.dropout(self.linear_prediction(x), training=self.training)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class SupCL(nn.Module):
    def __init__(self, dataset, args, device, augmentor):
        super(SupCL, self).__init__()
        self.device = device

        self.encoder = GraphLambdaLayer(dim=dataset.num_features, args=args)
        # self.encoder = GraphCNN(input_dim=dataset.num_features, hidden_dim=args.hidden, device=self.device)
        self.contrast_model = SupervisedContrast(device=self.device, tau=1, base_tau=1)

        self.augmentor = augmentor
        self.num_hops = args.num_hops
        self.max_freqs = args.max_freqs
        self.read = AvgReadout()
        # self.disc = Discriminator(args.hidden)
        self.resample_count = args.resample_count
        self.dataset_name = args.dataset
        self.MLP =  nn.Sequential(nn.BatchNorm1d(dataset.num_features, affine=False),
                    nn.Linear(dataset.num_features, 300),
                    nn.Dropout(p=args.dropout),
                    nn.ELU(),
                    # nn.BatchNorm1d(300, affine=False),
                    nn.Linear(300, 256),
                    nn.Dropout(p=args.dropout),
                    nn.ELU(),
                    # nn.BatchNorm1d(256, affine=False),
                    nn.Linear(256, args.hidden),
                    nn.Dropout(p=args.dropout),
                    nn.ELU())
        self.MLP.apply(init_weights)
        self.gate1 = nn.Linear(args.hidden * 2, args.hidden)
        self.gate2 = nn.Linear(args.hidden * 2, args.hidden)
        self.sigmoid = nn.Sigmoid()
        self.PE_train = args.PE_train
        if self.PE_train:
            self.adj = to_dense_adj(dataset.data.edge_index).squeeze()

    def forward(self, x, edge_index, y, train_mask):

        # g_1 = self.read(h_1, None).unsqueeze(0)
        # g_1 = g_1.repeat(h_1.shape[0],1)
        # import pdb;pdb.set_trace()
        if self.PE_train:
            UU,SS,VV = torch.svd_lowrank(self.adj, q=self.max_freqs)
            EigVals_1, EigVecs_1 = SS, VV
            EigVals_1 = EigVals_1.unsqueeze(0)
            EigVals_1 = EigVals_1.repeat(EigVecs_1.shape[0], 1).unsqueeze(2)
            # import pdb; pdb.set_trace()

            # try:
            #     with open(str(self.dataset_name) + '_eigvals_ori_high.pickle', 'rb') as f:
            #         EigVals_1 = pickle.load(f)
            #     with open(str(self.dataset_name) + '_eigvecs_ori_high.pickle', 'rb') as f2:
            #         EigVecs_1 = pickle.load(f2)
            # except:
            #     EigVals_1, EigVecs_1 = laplace_decomp(x, edge_index, self.max_freqs)
            #     with open(str(self.dataset_name) + '_eigvals_ori_high.pickle', 'wb') as f:
            #         pickle.dump(EigVals_1, f)
            #     with open(str(self.dataset_name) + '_eigvecs_ori_high.pickle', 'wb') as f2:
            #         pickle.dump(EigVecs_1, f2)
            # import pdb;pdb.set_trace()
            h_1 = self.encoder(x, edge_index, EigVals_1, EigVecs_1)
        else:
            h_1 = self.encoder(x, edge_index)
        h_2 = self.MLP(x)
        product = torch.cat((h_1, h_2), 1)
        h_1 = h_1 * self.sigmoid(self.gate1(product))
        h_2 = h_2 * self.sigmoid(self.gate2(product))
        h_1 = h_1 + h_2
        # h_1 = torch.cat((h_1, h_2), 1)
        # contrastive mask of shape [num_nodes, num_nodes], mask_{i,j}=1 if sample j has the same class as sample i.
        mask = torch.eq(y, y.unsqueeze(dim=1)).float()#.to('cpu')
        # loss = self.contrast_model(anchor=g_1, contrast=h_1, mask=mask)
        loss = self.contrast_model(anchor=h_1[train_mask], contrast=h_1[train_mask], mask=mask)

        return loss

class SchemaCL(nn.Module):
    def __init__(self, dataset, args, device, augmentor):
        super(SchemaCL, self).__init__()
        self.device = device

        self.encoder = GraphLambdaLayer(dim=dataset.num_features, args=args)
        self.contrast_model = SingleBranchContrast(loss=L.InfoNCE(tau=1.0), mode='G2L')

        self.augmentor = augmentor
        self.num_hops = args.num_hops
        self.max_freqs = args.max_freqs
        self.read = AvgReadout()
        # self.disc = Discriminator(args.hidden)
        self.resample_count = args.resample_count
        self.dataset_name = args.dataset
        self.PE_train = args.PE_train

    def forward(self, x, edge_index):

        # aug1, aug2 = self.augmentor
        # x1, edge_index1, _ = aug1(x, edge_index, None)
        # x2, edge_index2, _ = aug2(x, edge_index, None)
        x2, edge_index2, _ = self.augmentor(x, edge_index, None)
        if self.PE_train:
            try:
                with open(str(self.dataset_name) + '_eigvals_ori.pickle', 'rb') as f:
                    EigVals_1 = pickle.load(f)
                with open(str(self.dataset_name) + '_eigvecs_ori.pickle', 'rb') as f2:
                    EigVecs_1 = pickle.load(f2)
            except:

                EigVals_1, EigVecs_1 = laplace_decomp(x, edge_index, self.max_freqs)
                with open(str(self.dataset_name) + '_eigvals_ori.pickle', 'wb') as f:
                    pickle.dump(EigVals_1, f)
                with open(str(self.dataset_name) + '_eigvecs_ori.pickle', 'wb') as f2:
                    pickle.dump(EigVecs_1, f2)

            EigVals_2, EigVecs_2 = laplace_decomp(x2, edge_index2, self.max_freqs)

            h_1 = self.encoder(x, edge_index, EigVals_1, EigVecs_1)
            g_1 = self.read(h_1, None).unsqueeze(0)
            h_2 = self.encoder(x2, edge_index2, EigVals_2, EigVecs_2)
        else:
            h_1 = self.encoder(x, edge_index)
            g_1 = self.read(h_1, None).unsqueeze(0)
            h_2 = self.encoder(x2, edge_index2)
        loss = self.contrast_model(h=h_1, g=g_1, hn=h_2)

        return loss
