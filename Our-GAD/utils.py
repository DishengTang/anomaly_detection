from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import os.path as osp
from scipy.io import loadmat
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from torch_geometric.datasets import BitcoinOTC, UPFD
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_undirected, get_laplacian, to_dense_adj, dense_to_sparse

import torch
import torch.nn.functional as F
import pickle
import numpy as np


class MyDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None): # split='train',

        self.root = root
        self.name = name
        super(MyDataset, self).__init__(root, transform, pre_transform)

        # assert split in ['train', 'val', 'test']
        # path = self.processed_paths[['train', 'val', 'test'].index(split)]

        self.data, self.slides = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['data.mat', 'homo_adjlists.pickle']

    def download(self):
        # Download to self.raw_dir
        pass

    @property
    def processed_file_names(self):
        # return ['train.pt', 'val.pt', 'test.pt']
        return ['processed_data.pt']

    def process(self):
        # load the preprocessed adj_lists
        if self.name in ['YelpChi', 'Amazon']:
            with open(osp.join(self.raw_dir,'homo_adjlists.pickle'), 'rb') as file:
                homo = pickle.load(file)
            data_file = loadmat(osp.join(self.raw_dir, 'data.mat'))
            labels = data_file['label'].flatten()
            features = data_file['features'].todense().A

            x = torch.tensor(features).type(torch.FloatTensor)
            y = torch.tensor(labels).type(torch.LongTensor)
            data_list = [random_planetoid_splits(x, y, homo)]
        elif self.name in ['alpha','otc','epinions']:
            # extracting edge index and labels from files
            with open(osp.join(self.root, self.name, f'{self.name}_network.csv'), 'r') as file:
                edge1, edge2 = [], []
                for line in file:
                    line = line.strip().split(",")
                    node1, node2 = int(line[0]), int(line[1])
                    edge1.append(node1)
                    edge2.append(node2)
            edge_index = to_undirected(torch.from_numpy(np.array([edge1, edge2])))
            goodusers, badusers = [], []
            with open(osp.join(self.root, self.name, f'{self.name}_gt.csv'), 'r') as file:
                for line in file:
                    line = line.strip().split(",")
                    if line[1] == "-1":
                        badusers.append(int(line[0]))
                    else:
                        goodusers.append(int(line[0]))
            # making edge_indices and labels correspond to actual number of nodes
            node_ids = goodusers.copy()
            node_ids.extend(badusers)
            # edge_choose = []
            # # first remove some edges not in node_ids
            # for i in range(edge_index.shape[1]):
            #     if edge_index[0][i] in node_ids and edge_index[1][i] in node_ids:
            #         edge_choose.append(i)
            # edge_index = edge_index[:, edge_choose]

            num_nodes = len(torch.unique(edge_index[0]))
            uniq1 = torch.unique(edge_index[0])
            uniq2 = torch.Tensor(range(0, num_nodes,1))
            sorted = {uniq1[i].item(): uniq2[i] for i in range(num_nodes)}

            # comment: there are some labeled users in epinions dataset that are not connected with any other user so I delete them from labeled users
            i = 0
            while i < len(goodusers):
                if goodusers[i] in sorted:
                    goodusers[i] = int(sorted[goodusers[i]].item())
                    i += 1
                else:
                    goodusers.pop(i)
            i = 0
            while i < len(badusers):
                if badusers[i] in sorted:
                    badusers[i] = int(sorted[badusers[i]].item())
                    i += 1
                else:
                    badusers.pop(i)

            for i in range(edge_index[0].size()[0]):
                # update the node id in edge_index
                edge_index[0][i] = sorted[edge_index[0][i].item()]
                edge_index[1][i] = sorted[edge_index[1][i].item()]
            # edge_index = add_self_loops(edge_index) # perhaps need to add self loops?
            # split
            trn_rate, val_rate = 0.7, 0.1 # as in GCCAD
            indices = [torch.LongTensor(goodusers), torch.LongTensor(badusers)]
            train_index = torch.cat([i[:int(trn_rate * len(i))] for i in indices], dim=0)
            val_index = torch.cat([i[int(trn_rate * len(i)): int((trn_rate + val_rate) * len(i))] for i in indices], dim=0)
            test_index = torch.cat([i[int((trn_rate + val_rate) * len(i)):] for i in indices], dim=0)
            # preparing y
            # updated_node_ids = goodusers.copy()
            # updated_node_ids.extend(badusers)
            # num_nodes = len(updated_node_ids)

            good_mask = index_to_mask(torch.LongTensor(goodusers), size=num_nodes)
            bad_mask = index_to_mask(torch.LongTensor(badusers), size=num_nodes)

            y = torch.zeros(num_nodes)
            y[good_mask] = 0
            y[bad_mask] = 1
            y = y.to(torch.int64)
            # preparing x
            if self.name in ['alpha','otc']:  # 1) top 256 eigenvectors of the laplacian (OOM on epinions)
                L = get_laplacian(edge_index, normalization = 'sym')
                L = to_dense_adj(edge_index = L[0], edge_attr = L[1]).squeeze()
                D, Q = torch.linalg.eigh(L)
                x = Q[:, (num_nodes - 256):]
            else: # 2) random
                x = torch.rand(num_nodes, 256)
            # combining data
            data = Data(x=x, y=y, edge_index=edge_index)

            data.train_mask =  index_to_mask(train_index, size=data.num_nodes)
            data.val_mask =  index_to_mask(val_index, size=data.num_nodes)
            data.test_mask =  index_to_mask(test_index, size=data.num_nodes)

            data_list = [data]

        for i, path in enumerate(self.processed_paths):
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_filter is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            torch.save(self.collate([data_list[i]]), path)


def metric_results(y_true, y_logits):
    all_results = {}
    y_pred = y_logits.argmax(axis=1)
    all_results['auc'] = roc_auc_score(y_true, y_logits[:,1])
    all_results['ap'] = average_precision_score(y_true, y_logits[:,1])
    all_results['acc'] = accuracy_score(y_true, y_pred)
    all_results['recall'] = recall_score(y_true, y_pred, average="macro")
    all_results['F1_macro'] = f1_score(y_true, y_pred, average="macro")
    all_results['F1_micro'] = f1_score(y_true, y_pred, average="micro")
    return all_results

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def get_edge_index(edge_index, mask_index):
    adj = to_dense_adj(edge_index).squeeze(0)
    new_adj = adj[:, mask_index][mask_index, :]
    new_edge_index, _ = dense_to_sparse(new_adj)
    return new_edge_index


def random_planetoid_splits(x,y, neighbor_set, trn_rate=0.4, val_rate=0.01):
    indices = []
    num_classes = len(torch.unique(y))
    for i in range(num_classes):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:int(trn_rate * len(i))] for i in indices], dim=0)
    val_index = torch.cat([i[int(trn_rate * len(i)): int((trn_rate + val_rate) * len(i))] for i in indices], dim=0)
    test_index = torch.cat([i[int((trn_rate + val_rate) * len(i)):] for i in indices], dim=0)
    edge_index = np.array([(j, i) for i in neighbor_set for j in neighbor_set[i]]).T
    # train_edge_index, val_edge_index, test_edge_index = [],[],[]
    # for i, neig in neighbor_set.items():
    #     if i in train_index:
    #         for j in neig:
    #             if j in train_index:
    #                 train_edge_index.append(list([j,i]))
    #     elif i in val_index:
    #         for j in neig:
    #             if j in val_index:
    #                 val_edge_index.append(list([j,i]))
    #     else:
    #         for j in neig:
    #             if j in test_index:
    #                 test_edge_index.append(list([j,i]))

    # train_data = Data(x=x[train_index], y=y[train_index], edge_index=torch.tensor(edge_index).type(torch.LongTensor))
    # val_data = Data(x=x[val_index], y=y[val_index], edge_index=torch.tensor(edge_index).type(torch.LongTensor))
    # test_data = Data(x=x[test_index], y=y[test_index], edge_index=torch.tensor(edge_index).type(torch.LongTensor))
    edge_index = torch.tensor(edge_index).type(torch.LongTensor)
    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask =  index_to_mask(train_index, size=data.num_nodes)
    data.val_mask =  index_to_mask(val_index, size=data.num_nodes)
    data.test_mask =  index_to_mask(test_index, size=data.num_nodes)

    data.train_edge_index = get_edge_index(edge_index, train_index)
    data.val_edge_index = get_edge_index(edge_index, val_index)
    data.test_edge_index = get_edge_index(edge_index, test_index)
    return data

def LoadDataSet(data_name):
    '''
    return PyG-formed datasets, train_dataset, val_dataset, test_dataset
    for each dataset .data includes x, y, edge_index
    '''
    root_path = osp.join(osp.expanduser('~'), 'anomaly/datasets/')
    if data_name in ['Amazon', 'YelpChi','alpha', 'otc', 'epinions']:
        # train_dataset = MyDataset(root=root_path, name=data_name, split='train')
        # val_dataset = MyDataset(root=root_path, name=data_name, split='val')
        # test_dataset = MyDataset(root=root_path, name=data_name, split='test')
        dataset = MyDataset(root=root_path, name=data_name)

    # elif data_name == 'OTC':
    #     otc_path = root_path + '/Bitcoin-OTC/'
    #     dataset = BitcoinOTC(root=otc_path)
        ## need to split dataset

    else:
        data_name = data_name.split('-')
        if data_name[0] == 'UPFD':
            upfd_path = root_path + '/UPFD/'

            upfd_data_feature = ['profile', 'spacy', 'bert', 'content']
            # data_name[1] choose from ['politifact', 'gossipcop']
            train_dataset = UPFD(upfd_path, data_name[1], upfd_data_feature[0], 'train', ToUndirected())
            val_dataset = UPFD(upfd_path, data_name[1], upfd_data_feature[0], 'val', ToUndirected())
            test_dataset = UPFD(upfd_path, data_name[1], upfd_data_feature[0], 'test', ToUndirected())
            return train_dataset, val_dataset, test_dataset
        else:
            print('dataset name not support')
            return

    return dataset


def laplace_decomp(x, edge_index, max_freqs):
    # Laplacian
    n = x.shape[0]
    L = to_dense_adj(get_laplacian(edge_index.cpu(), normalization='sym')[0]) # L = I - D^{-1/2}AD^{-1/2}


    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.numpy())


    EigVals, EigVecs = EigVals.squeeze(0), EigVecs.squeeze(0)
    # EigVals, EigVecs = EigVals[:max_freqs], EigVecs[:, :max_freqs] # the top low frequency
    EigVals, EigVecs = EigVals[-max_freqs:], EigVecs[:, -max_freqs:]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)

    if n < max_freqs:
        EigVecs = F.pad(EigVecs, (0, max_freqs - n), value=float('nan'))

    if EigVecs.shape[0] < n:  # the real number of nodes is less than x.shape[0], pad
        EigVecs = F.pad(EigVecs, (0, 0, 0, n - EigVecs.shape[0]), value=float('nan'))
    # Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals))))  # Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative

    if n < max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs - n), value=float('nan')).unsqueeze(0)
    else:
        EigVals = EigVals.unsqueeze(0)

    # Save EigVals node features
    EigVals = EigVals.repeat(n, 1).unsqueeze(2)

    return EigVals, EigVecs

from torch_sparse import SparseTensor
from torch_scatter import scatter_mean

def homophily_(edge_index, y, mask, batch=None):
    y = y.squeeze(-1) if y.dim() > 1 else y

    if isinstance(edge_index, SparseTensor):
        col, row, _ = edge_index.coo()
    else:
        row, col = edge_index
    mask_row = torch.Tensor([k for k in row if mask[k]]).type(torch.int64).to(device=row.device)
    mask_col = torch.Tensor([k for k in col if mask[k]]).type(torch.int64).to(device=row.device)
    out = torch.zeros(mask_row.size(0), device=row.device)
    out[y[mask_row] == y[mask_col]] = 1.
    out = scatter_mean(out, mask_col, 0, dim_size=y.size(0))
    if batch is None:
        return float(out.mean())
    else:
        return scatter_mean(out, batch, dim=0)