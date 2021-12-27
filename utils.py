from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import os.path as osp
from scipy.io import loadmat
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from torch_geometric.datasets import BitcoinOTC, UPFD
from torch_geometric.transforms import ToUndirected

import torch
import pickle
import numpy as np


class MyDataset(InMemoryDataset):
    def __init__(self, root, name, split='train', transform=None, pre_transform=None):

        self.root = root
        self.name = name
        super(MyDataset, self).__init__(root, transform, pre_transform)

        assert split in ['train', 'val', 'test']
        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.data, self.slides = torch.load(path)

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
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self):
        # load the preprocessed adj_lists
        with open(osp.join(self.raw_dir,'homo_adjlists.pickle'), 'rb') as file:
            homo = pickle.load(file)
        data_file = loadmat(osp.join(self.raw_dir, 'data.mat'))
        labels = data_file['label'].flatten()
        features = data_file['features'].todense().A

        x = torch.tensor(features).type(torch.FloatTensor)
        y = torch.tensor(labels).type(torch.LongTensor)


        data_list = random_planetoid_splits(x, y, homo)
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


def random_planetoid_splits(x,y,neighbor_set, trn_rate=0.4, val_rate=0.01):

    indices = []
    num_classes = len(torch.unique(y))
    for i in range(num_classes):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:int(trn_rate * len(i))] for i in indices], dim=0)
    val_index = torch.cat([i[int(trn_rate * len(i)): int((trn_rate + val_rate) * len(i))] for i in indices], dim=0)
    test_index = torch.cat([i[int((trn_rate + val_rate) * len(i)):] for i in indices], dim=0)

    train_edge_index, val_edge_index, test_edge_index = [],[],[]
    for i, neig in neighbor_set.items():
        if i in train_index:
            for j in neig:
                if j in train_index:
                    train_edge_index.append(list([j,i]))
        elif i in val_index:
            for j in neig:
                if j in val_index:
                    val_edge_index.append(list([j,i]))
        else:
            for j in neig:
                if j in test_index:
                    test_edge_index.append(list([j,i]))

    train_data = Data(x=x[train_index], y=y[train_index], edge_index=torch.tensor(np.array(train_edge_index).T).type(torch.LongTensor))
    val_data = Data(x=x[val_index], y=y[val_index], edge_index=torch.tensor(np.array(val_edge_index).T).type(torch.LongTensor))
    test_data = Data(x=x[test_index], y=y[test_index], edge_index=torch.tensor(np.array(test_edge_index).T).type(torch.LongTensor))

    return [train_data, val_data, test_data]

def LoadDataSet(data_name):
    '''
    return PyG-formed datasets, train_dataset, val_dataset, test_dataset
    for each dataset .data includes x, y, edge_index
    '''
    root_path = osp.join(osp.expanduser('~'), 'datasets/')
    if data_name == 'Amazon' or 'YelpChi':
        train_dataset = MyDataset(root=root_path, name=data_name, split='train')
        val_dataset = MyDataset(root=root_path, name=data_name, split='val')
        test_dataset = MyDataset(root=root_path, name=data_name, split='test')


    elif data_name == 'OTC':
        otc_path = root_path + '/Bitcoin-OTC/'
        dataset = BitcoinOTC(root=otc_path)
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
        else:
            print('dataset name not support')
            return

    return train_dataset, val_dataset, test_dataset