# %%
import argparse
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tqdm import tqdm
from utils import LoadDataSet, metric_results, preprocess_neighbors_sumavepool
from model.GNN_models import *

from sklearn.cluster import KMeans # for clustering

import TableIt # for beatiful tables output

# PGD attack model
class AttackPGD(nn.Module):
    def __init__(self, model, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']

    def forward(self, feature, adj, cluster_info, num_cluster):
        feats_adv = feature.clone().detach()
        if self.rand:
            feats_adv = feats_adv + torch.zeros_like(feats_adv).uniform_(-self.epsilon, self.epsilon)
        idx = np.random.permutation(feature.shape[0])
        shuf_feats = feature[idx, :]
        for i in range(self.num_steps):
            feats_adv.requires_grad_()
            with torch.enable_grad():
                loss = self.model(feats_adv, shuf_feats, adj, None, None, None, cluster_info, num_cluster)
            grad_feats_adv = torch.autograd.grad(loss, feats_adv)
            feats_adv = feats_adv.detach() + self.step_size * torch.sign(grad_feats_adv[0].detach())
            feats_adv = torch.min(torch.max(feats_adv, feature - self.epsilon), feature + self.epsilon)
            # import pdb;pdb.set_trace()
            # feats_adv = torch.clamp(feats_adv, 0, 1)
        return feats_adv

def run(args, dataset, Net):
    
    print('Pretraining...')
    # fragment for DCI and DGI (can be moved to another function/place if needed)
    if args.net in ['DCI', 'DGI', 'DGI_MLP']:
        data = dataset.data.cpu()
        # bn = nn.BatchNorm1d(data.x.shape[1], affine=False)
        # import pdb;pdb.set_trace()
        # data.x = bn(data.x)
        # data.x = F.normalize(data.x)
        edge_index = data.edge_index
        feats = data.x 
        nb_nodes = feats.size()[0] 
        input_dim = feats.size()[1]
        idx = np.random.permutation(nb_nodes)
        shuf_feats = feats[idx, :]
        
        if args.net == 'DCI':
            kmeans = KMeans(n_clusters=args.num_cluster, random_state=0).fit(feats)
            ss_label = kmeans.labels_
            cluster_info = [list(np.where(ss_label==i)[0]) for i in range(args.num_cluster)]
            
        adj = preprocess_neighbors_sumavepool(torch.LongTensor(edge_index), nb_nodes, device)
        feats = torch.FloatTensor(feats).to(device)
        shuf_feats = torch.FloatTensor(shuf_feats).to(device)
        if args.net == 'DCI':
            model_pretrain = DCI(args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, args.neighbor_pooling_type, device, dataset, args).to(device)
            if args.adv:
                attack = AttackPGD(model_pretrain, args.config)
            if args.training_scheme == 'decoupled':
                optimizer_train = torch.optim.Adam(model_pretrain.parameters(), lr=args.lr)
                for epoch in tqdm(range(1, args.epochs + 1)):
                    if args.adv:
                        attack.train()
                        feats_adv = attack(feats, adj, cluster_info, args.num_cluster)
                    model_pretrain.train()
                    loss_pretrain = model_pretrain(feats, shuf_feats, adj, None, None, None, cluster_info, args.num_cluster)
                    if args.adv:
                        loss_pretrain = loss_pretrain + model_pretrain(feats_adv, shuf_feats, adj, None, None, None, cluster_info, args.num_cluster)
                    # model_pretrain.train()
                    # loss_pretrain = model_pretrain(feats, shuf_feats, adj, None, None, None, cluster_info, args.num_cluster)
                    if optimizer_train is not None:
                        optimizer_train.zero_grad()
                        loss_pretrain.backward()         
                        optimizer_train.step()
                    # re-clustering
                    if epoch % args.recluster_interval == 0 and epoch < args.epochs:
                        model_pretrain.eval()
                        # emb = model_pretrain.get_emb(data)
                        emb = model_pretrain.get_emb(feats, adj)
                        kmeans = KMeans(n_clusters=args.num_cluster, random_state=0).fit(emb.detach().cpu().numpy())
                        ss_label = kmeans.labels_
                        cluster_info = [list(np.where(ss_label==i)[0]) for i in range(args.num_cluster)]
        
            print('Pre-training of DCI is done!')
        
        if args.net in ['DGI','DGI_MLP']:
            batch_size = 1
            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(device)
            
            model_pretrain = DGI(args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, args.neighbor_pooling_type, device, dataset, args).to(device)
            optimizer_train = torch.optim.Adam(model_pretrain.parameters(), lr=args.lr)
            model_pretrain.train()
            for _ in range(1, args.epochs + 1):
                loss_pretrain = model_pretrain(feats, shuf_feats, adj, None, None, None, lbl)
                if optimizer_train is not None:
                    optimizer_train.zero_grad()
                    loss_pretrain.backward()         
                    optimizer_train.step()
            
            
            print('Pre-training of DGI is done!')
        
        
    def train(model, optimizer, dataset, batch_train=False):
        
        if not batch_train:
            data = dataset.data
            if args.cuda:
                data = data.to(device)
            model.train()
            optimizer.zero_grad()
            out = model(data)[data.train_mask]
            loss = model.m_loss()(out, data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        # mini-batch training
        else:
            loss = 0
            loader = DataLoader(dataset, batch_size=args.batchsize)
            model.train()
            for data in loader:
                if args.cuda:
                    data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                b_loss = model.m_loss()(out, data.y[data.train_mask])
                b_loss.backward()
                optimizer.step()
                loss += b_loss.item()


    def test(model, data, mask):
        model.eval()   
        y_logits = model.to_prob(data)
        loss = model.m_loss()(model(data)[mask], data.y[mask])
        results = metric_results(y_true=data.y[mask].data.cpu().numpy(), y_logits=y_logits[mask].data.cpu().numpy())
        return loss, results

    if args.net == 'DGI_MLP':
        h = model_pretrain.get_emb(data)
        data.x = h
        model = Net(dataset, args)
    else:
        model = Net(dataset, args)
    if args.cuda:
        model = model.to(device)
    if args.net in ['DGI', 'DGI']:
        pretrained_dict = model_pretrain.state_dict()
        model_dict = model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    for epoch in tqdm(range(args.epochs)):
        train(model, optimizer, dataset, args.batch_train)
        # train_loss, train_res = test(model, train_dataset.data)
        val_loss, val_res = test(model, dataset.data, dataset.data.val_mask)
        test_loss, test_res = test(model, dataset.data, dataset.data.test_mask)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_res = val_res
            best_test_res = test_res
            best_epoch = epoch
    return best_epoch, best_test_res


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--train_rate', type=float, default=0.4)
parser.add_argument('--val_rate', type=float, default=0.1)
parser.add_argument('--adv', type=int, default=0)
parser.add_argument('--batch_train', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--dataset', type=str, choices= ['YelpChi', 'Amazon', 'OTC', 'UPFD'], default='Amazon')
parser.add_argument('--net', type=str, choices=['CARE-GNN', 'our-GAD', 'GPR-GNN', 'GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'DGI', 'DCI', 'DGI_MLP'], default='DCI')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--RPMAX', type=int, default=1, help='repeat count')
parser.add_argument('--K', type=int, default=8)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--dprate', type=float, default=0.0)
parser.add_argument('--Init', type=str,
                    choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                    default='PPR')
parser.add_argument('--Gamma', type=str, default=None)
parser.add_argument('--heads', default=1, type=int)
parser.add_argument('--output_heads', default=1, type=int)
# args for DCI and DGI:
parser.add_argument('--final_dropout', type=float, default=0.5,
                    help='final layer dropout (default: 0.5)')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--num_cluster', type=int, default=2,   
                    help='number of clusters (default: 2)') 
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers (default: 2)')
parser.add_argument('--num_mlp_layers', type=int, default=2,
                    help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='number of hidden units (default: 128)')
parser.add_argument('--neighbor_pooling_type', type=str, default="average", choices=["sum", "average"],
                    help='Pooling for over neighboring nodes: sum or average')
parser.add_argument('--training_scheme', type=str, default="decoupled", choices=["decoupled", "joint"],
                    help='Training schemes: decoupled or joint')
parser.add_argument('--recluster_interval', type=int, default=20,
                    help='the interval of reclustering (default: 20)')

args = parser.parse_args(['--dataset', 'YelpChi'])
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda:" + str(args.device)) if args.cuda else torch.device("cpu") # this is more flexible

print(f'Loading {args.dataset}...')

dataset = LoadDataSet(args.dataset)
# import pdb; pdb.set_trace()
gnn_name = args.net
if gnn_name == 'GPR-GNN':
    Net = GPRGNN
elif gnn_name == 'GAT':
    Net = GAT_Net
elif gnn_name == 'APPNP':
    Net = APPNP_Net
elif gnn_name == 'ChebNet':
    Net = ChebNet
elif gnn_name == 'JKNet':
    Net = GCN_JKNet
elif gnn_name == 'GCN':
    Net = GCN_Net
elif gnn_name == 'DGI':
    Net = Classifier
elif gnn_name == 'DCI':
    Net = Classifier
elif gnn_name == 'DGI_MLP':
    Net = Classifier_MLP

# %%
import torch_cluster
from torch_cluster import random_walk
from torch_sparse import SparseTensor
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
# %%
# data = dataset.data
# subgraph_size=3; walk_length=4
# def subgraph_rw(data, subgraph_size=3, walk_length=1):
#     (row, col), N = data.edge_index, data.num_nodes
#     all_nodes = torch.arange(N)
#     all_nodes = all_nodes.repeat_interleave(subgraph_size)
#     rw = random_walk(row, col, all_nodes, walk_length)
#     if not isinstance(rw, torch.Tensor):
#         rw = rw[0]
#     subv = []
#     for i in tqdm(range(N)):
#         trace = rw[i*subgraph_size:(i+1)*subgraph_size, 1:]
#         subv.append(torch.unique(trace.T.flatten()).tolist()) # transpose makes the 1-st neighbor ranking ahead of higher-order
#         retry_time = 0
#         while len(subv[i]) < subgraph_size: # if it does not have at least 3 neighbors, reach for higher order neighbors
#             rw = random_walk(row, col, torch.tensor([i]), walk_length=subgraph_size+1)
#             subv[i] = torch.unique(rw[:, 1:].T.flatten()).tolist()
#             retry_time += 1
#             if (len(subv[i]) < subgraph_size) and (retry_time >1):
#                 subv[i] = (subv[i] * subgraph_size)
#         subv[i] = subv[i][:subgraph_size]
#     return subv
# subv = subgraph_rw(data, subgraph_size=4, walk_length=4)
# %%
# subgraph_size = 5
# subgraph_size=3; walk_length=4
# def subgraph_rw_fast(data, subgraph_size=3):
#     (row, col), N = data.edge_index, data.num_nodes
#     adj = to_dense_adj(data.edge_index).squeeze()
#     adj.fill_diagonal_(0)
#     adj_2 = torch.mm(adj, adj)
#     subgraphs = []
#     for n in tqdm(range(N)):
#         subgraphs.append(adj[n,:].nonzero().flatten().tolist()[:subgraph_size])
#         if len(subgraphs[n]) < subgraph_size: # if it does not have at least 3 neighbors, reach for higher order neighbors
#             rw = random_walk(row, col, torch.tensor([n]*subgraph_size), walk_length=subgraph_size)
#             subgraphs[n] = torch.unique(rw[:, 1:].T.flatten()).tolist()
#             if (len(subgraphs[n]) < subgraph_size):
#                 subgraphs[n] = (subgraphs[n] * subgraph_size)
#         subgraphs[n] = subgraphs[n][:subgraph_size]
#     return subgraphs
# subgraphs = subgraph_rw_fast(data, subgraph_size=4)

# %%
import copy
from torch_geometric.utils import remove_self_loops
data = dataset.data
def subgraph_rw_fast(data, first_neighbor=None, subgraph_size=3):
    edge_index = remove_self_loops(data.edge_index)[0]
    (row, col), N = edge_index, data.num_nodes
    if first_neighbor is not None:
        subgraphs = copy.deepcopy(first_neighbor)
        n_size = torch.tensor([len(s) for s in subgraphs])
    else:
        adj = to_dense_adj(data.edge_index).squeeze()
        adj.fill_diagonal_(0)
        n_size = torch.zeros(N)
        subgraphs = []
        for n in range(N):
            nonzeros = adj[n,:].nonzero().flatten()
            subgraphs.append(nonzeros[torch.randperm(len(nonzeros))].tolist()[:subgraph_size])
            n_size[n] = len(subgraphs[n])
        first_neighbor = copy.deepcopy(subgraphs)
    # import pdb;pdb.set_trace()
    node_idx = torch.where(n_size<subgraph_size)[0]
    rw_node_idx = node_idx.repeat_interleave(subgraph_size)
    rw = random_walk(row, col, rw_node_idx, walk_length=subgraph_size)
    for i in tqdm(range(len(node_idx))):
        trace = rw[i*subgraph_size:(i+1)*subgraph_size, 1:]
        subgraphs[node_idx[i]] = torch.unique(trace[:, 1:].T.flatten()).tolist()
        if (len(subgraphs[node_idx[i]]) < subgraph_size):
            subgraphs[node_idx[i]] = (subgraphs[node_idx[i]] * subgraph_size)
        subgraphs[node_idx[i]] = subgraphs[node_idx[i]][:subgraph_size]
    return first_neighbor, subgraphs
first_neighbor = None
first_neighbor, subgraphs = subgraph_rw_fast(data, first_neighbor=first_neighbor, subgraph_size=4)
# first_neighbor, subgraphs = subgraph_rw_fast(data, first_neighbor=first_neighbor, subgraph_size=4)
# %%
