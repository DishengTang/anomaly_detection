from torch_geometric.data import Data
import os.path as osp
from scipy.io import loadmat
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from torch_geometric.datasets import BitcoinOTC, UPFD
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import DataLoader
import torch

class MyDataset:
    def __init__(self, data):
        self.data = data
        self.num_features, self.num_classes = data.x.shape[1], len(torch.unique(data.y))

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


def random_planetoid_splits(x,y,neighbor_set, trn_rate=0.4, val_rate=0.1):

    indices = []
    num_classes = len(torch.unique(data.y))
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:int(trn_rate * len(i))] for i in indices], dim=0)
    val_index = torch.cat([i[int(trn_rate * len(i)): int((trn_rate + val_rate) * len(i))] for i in indices], dim=0)
    test_index = torch.cat([i[int((trn_rate + val_rate) * len(i)):] for i in indices], dim=0)


    train_edge_index, val_edge_index, test_edge_index = [],[],[]
    for i in neighbor_set:
        for j in neighbor_set[i]:
            if i in train_index and j in train_index:
                train_edge_index.append((j,i))
            elif i in val_index and j in val_index:
                val_edge_index.append((j,i))
            elif i in test_index and j in test_index:
                test_edge_index.append((j,i))

    train_data = Data(x=x[train_index], y=y[train_index], edge_index=torch.tensor(np.array(train_edge_index)).type(torch.LongTensor))
    val_data = Data(x=x[val_index], y=y[val_index], edge_index=torch.tensor(np.array(val_edge_index)).type(torch.LongTensor))
    test_data = Data(x=x[testindex], y=y[test_index], edge_index=torch.tensor(np.array(test_edge_index)).type(torch.LongTensor))
    train_dataset = MyDataset(train_data)
    val_dataset = MyDataset(val_data)
    test_dataset = MyDataset(test_data)

    return train_dataset, val_dataset, test_dataset

def LoadDataSet(data_name):
    '''
    return PyG-formed datasets, train_dataset, val_dataset, test_dataset
    for each dataset .data includes features, labels, edge_index; .num_feature, .num_classes
    '''
    root_path = osp.join(osp.expanduser('~'), 'datasets/')
    if data_name == 'Amazon' or 'YelpChi':
        data_file = loadmat(root_path + data_name + '/' + data_name + '.mat')
        labels = data_file['label'].flatten()
        features = data_file['features'].todense().A
        # load the preprocessed adj_lists
        if data_name == 'YelpChi':
            with open(root_path + data_name + '/' + 'yelp_homo_adjlists.pickle', 'rb') as file:
                homo = pickle.load(file)
        else:
            with open(root_path + data_name + '/' + 'amz_homo_adjlists.pickle', 'rb') as file:
                homo = pickle.load(file)

        x = torch.tensor(features).type(torch.FloatTensor)
        y = torch.tensor(labels).type(torch.LongTensor)
        # data = Data(x=, y=,
        #             edge_index=torch.tensor(edge_index).type(torch.LongTensor))
        train_dataset, val_dataset, test_dataset = random_planetoid_splits(x, y, homo)

    elif data_name == 'OTC':
        otc_path = root_path + '/Bitcoin-OTC/'
        dataset = BitcoinOTC(root=otc_path)
        ## need to split dataset

    elif data_name == 'UPFD':
        upfd_path = root_path + '/UPFD/'
        upfd_data_name = ['politifact', 'gossipcop']
        upfd_data_feature = ['profile', 'spacy', 'bert', 'content']

        train_dataset = UPFD(upfd_path, upfd_data_name[0], upfd_data_feature[0], 'train', ToUndirected())

    return train_dataset, val_dataset, test_dataset

# def _dataloader(data_name, dataset, batchsize=128, bool_shuffle=True):
#     if data_name == 'UPFD':
#         loader = DataLoader(dataset, batch_size=batchsize, shuffle=bool_shuffle)
#     elif data_name == 'Amazon' or data_name == 'YelpChi':
#         loader = MyDataLoader(dataset, batch_size=batchsize, shuffle=bool_shuffle)
