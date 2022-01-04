#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP

import torch_sparse

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='mean', **kwargs)  #aggr='add'
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))

        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(dataset.num_classes, args.hidden)) #
        torch.nn.init.xavier_uniform_(self.weight)
        self.softmax = torch.nn.Softmax(dim=1)

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def m_loss(self):
        return nn.CrossEntropyLoss()

    def to_prob(self,data):
        return self.softmax(self.forward(data))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.lin1(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)

        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        emb = F.relu(self.weight.mm(x.t()).t())
        return emb

class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout
        self.softmax = torch.nn.Softmax(dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def m_loss(self):
        return nn.CrossEntropyLoss()

    def to_prob(self,data):
        return self.softmax(self.forward(data))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.relu(x)


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 32, K=2)
        self.conv2 = ChebConv(32, dataset.num_classes, K=2)
        self.dropout = args.dropout
        self.softmax = torch.nn.Softmax(dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def m_loss(self):
        return nn.CrossEntropyLoss()

    def to_prob(self,data):
        return self.softmax(self.forward(data))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.relu(x)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout
        self.softmax = torch.nn.Softmax(dim=1)

    def m_loss(self):
        return nn.CrossEntropyLoss()

    def to_prob(self,data):
        return self.softmax(self.forward(data))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.relu(x)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout
        self.softmax = torch.nn.Softmax(dim=1)

    def m_loss(self):
        return nn.CrossEntropyLoss()

    def to_prob(self,data):
        return self.softmax(self.forward(data))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.relu(x)


class GCN_JKNet(torch.nn.Module):
    def __init__(self, dataset, args):
        in_channels = dataset.num_features
        out_channels = dataset.num_classes

        super(GCN_JKNet, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin1 = torch.nn.Linear(16, out_channels)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=16,
                                   num_layers=4
                                   )
        self.softmax = torch.nn.Softmax(dim=1)

    def m_loss(self):
        return nn.CrossEntropyLoss()

    def to_prob(self,data):
        return self.softmax(self.forward(data))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        return F.relu(x)
##############################
# DCI/DGI
##############################  

class GPRGNN_encoder(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN_encoder, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.hidden_dim)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    # def forward(self, data):
        # x, edge_index = data.x, data.edge_index
    def forward(self, x, edge_index):
        if edge_index.is_sparse:
            edge_index = edge_index.coalesce().indices()

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return x
                  
class DCI(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, negsamp_round, device, dataset, args):
        super(DCI, self).__init__()
        self.device = device
        self.negsamp_round = negsamp_round
        self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(hidden_dim, negsamp_round)
        self.gprgnn = GPRGNN_encoder(dataset, args)
        self.data = dataset.data.to(device)
        self.norm_layer = nn.BatchNorm1d(input_dim, affine=False)

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2, cluster_info, cluster_num):
        seq1 = self.norm_layer(seq1)
        seq2 = self.norm_layer(seq2)
        h_1 = self.gprgnn(seq1, adj)
        h_2 = self.gprgnn(seq2, adj)

        # h_1 = self.gprgnn(self.data)
        # self.data.x = seq2
        # h_2 = self.gprgnn(self.data)
        # self.data.x = seq1

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
            lbl_2 = torch.zeros(batch_size*self.negsamp_round, len(node_idx))
            # import pdb;pdb.set_trace()
            lbl = torch.cat((lbl_1, lbl_2), 0).to(self.device)
            # lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)

            ret = self.disc(c_block, h_1_block, h_2_block, samp_bias1, samp_bias2)
            loss_tmp = criterion(ret, lbl)
            loss += loss_tmp

        return loss / cluster_num

    # def get_emb(self, data):
    #     h_1 = self.gprgnn(data)
    def get_emb(self, seq, adj):
        h_1 = self.gprgnn(seq, adj)
        return h_1
        
class DGI(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device, dataset, args):
        super(DGI, self).__init__()
        self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
        self.gprgnn = GPRGNN_encoder(dataset, args)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(hidden_dim)
        self.data = dataset.data.to(device)

    def get_emb(self, data):
        h_1 = self.gprgnn(data)
        return h_1
    
    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2, lbl):
        criterion = nn.BCEWithLogitsLoss()
        
        h_1 = torch.unsqueeze(self.gprgnn(self.data), 0)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        self.data.x = seq2
        h_2 = torch.unsqueeze(self.gprgnn(self.data), 0)
        
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        loss = criterion(ret, lbl)
        
        self.data.x = seq1

        return loss

class Classifier(nn.Module):
    def __init__(self, dataset, args):
        super(Classifier, self).__init__()
        out_channels = dataset.num_classes
        device = torch.device("cuda:" + str(args.device)) if args.cuda else torch.device("cpu")
        self.gin = GraphCNN(args.num_layers, args.num_mlp_layers, dataset.data.x.size()[1], args.hidden_dim, args.neighbor_pooling_type, device)
        self.linear_prediction = nn.Linear(args.hidden_dim, out_channels)
        self.final_dropout = args.final_dropout
        self.gprgnn = GPRGNN_encoder(dataset, args)
        self.softmax = torch.nn.Softmax(dim=1)
        self.norm_layer = nn.BatchNorm1d(dataset.data.x.size()[1], affine=False)
        
    def m_loss(self):
        return nn.CrossEntropyLoss()

    def to_prob(self,data):
        return self.softmax(self.forward(data))
    
    def forward(self, data):
        # h_1 = self.gprgnn(data)
        h_1 = self.gprgnn(self.norm_layer(data.x), data.edge_index)
        score_final_layer = F.dropout(self.linear_prediction(h_1), self.final_dropout, training = self.training)
        return score_final_layer

class Classifier_MLP(nn.Module):
    def __init__(self, dataset, args):
        super(Classifier_MLP, self).__init__()
        out_channels = dataset.num_classes
        input_dim = dataset.data.x.size()[0]
         
        device = torch.device("cuda:" + str(args.device)) if args.cuda else torch.device("cpu")
        self.linear_prediction = nn.Linear(input_dim, out_channels)
        self.final_dropout = args.final_dropout
        self.softmax = torch.nn.Softmax(dim=1)
        
    def m_loss(self):
        return nn.CrossEntropyLoss()

    def to_prob(self, data):
        return self.softmax(self.forward(data))
    
    def forward(self, data):
        
        h = data.x
        score_final_layer = F.dropout(self.linear_prediction(h), self.final_dropout, training = self.training)
        return score_final_layer


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.negsamp_round = negsamp_round

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        scs = []
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        if s_bias1 is not None:
            sc_1 += s_bias1
        scs.append(sc_1)
        for neg in range(self.negsamp_round):
            if neg == 0:
                sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
            else:
                nb_nodes = h_mi.size()[0]
                idx = np.random.permutation(nb_nodes)
                sc_2 = torch.squeeze(self.f_k(h_mi[idx, :], c_x), 2)
            if s_bias2 is not None:
                sc_2 += s_bias2
            scs.append(sc_2)

        # logits = torch.cat((sc_1, sc_2), 1)
        logits = torch.cat(tuple(scs))

        return logits
        
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
        '''
            num_layers: number of layers in the neural networks
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None, classifier = False):
        ###pooling neighboring nodes and center nodes altogether  
        
        #If sum or average pooling
        if classifier == True:
            pooled = torch_sparse.spmm(Adj_block, m = h.size()[0], n = h.size()[0], matrix = h, value = torch.ones(Adj_block.size()[1]).to(self.device))
        else:
            pooled = torch.spmm(Adj_block, h)
        
        if self.neighbor_pooling_type == "average":
            #If average pooling
            if classifier == True:
                degree = torch_sparse.spmm(Adj_block, m = h.size()[0], n = h.size()[0], matrix = torch.ones((h.size()[0], 1)).to(self.device), value = torch.ones(Adj_block.size()[1]).to(self.device))
            else:
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
            
            pooled = pooled/degree

        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h

    
    def forward(self, feats, adj, classifier = False):
        h = feats
        for layer in range(self.num_layers):
            h = self.next_layer(h, layer, Adj_block = adj, classifier = classifier)

        return h
        
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)
            
#######################
