import argparse

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import homophily
from tqdm import tqdm
from utils import LoadDataSet, metric_results, homophily_
from model.GNN_models import *
from model.DCL_model import SchemaCL, SupCL, DCI
from model.clf_model import Classifier#, DCI_Classifier
from utils import laplace_decomp
import model.GCL.augmentors as Aug
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from numpy import linalg as LA
from sklearn.metrics.pairwise import cosine_similarity
import time
import pickle
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 13
# device = torch.device('cuda')

def remove_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]

def run(args, dataset, Net, model_pretrain=None):

    def train(model, optimizer, dataset, batch_train=False):

        if not batch_train:
            data = dataset.data
            if args.cuda:
                data = data.to(device)
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)[data.train_mask]
            loss = model.m_loss()(out, data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        return loss.detach().item()

    ###  inductive test
    # def test(model, data, mode):
    #     model.eval()
    #     if args.cuda:
    #         data = data.to(device)
    #     if mode == 'val':
    #         x = data.x[data.val_mask]
    #         edge_index = data.val_edge_index
    #         y = data.y[data.val_mask]
    #     else:
    #         x = data.x[data.test_mask]
    #         edge_index = data.test_edge_index
    #         y = data.y[data.test_mask]
    #
    #     y_logits = model.to_prob(x, edge_index)
    #     loss = model.m_loss()(model(x, edge_index), y)
    #     results = metric_results(y_true=y.data.cpu().numpy(), y_logits=y_logits.data.cpu().numpy())
    #     return loss, results

    ### transductive test
    def test(model, data, mask):
        model.eval()
        if args.cuda:
            data = data.to(device)

        y_logits = model.to_prob(data.x, data.edge_index)
        loss = model.m_loss()(model(data.x, data.edge_index)[mask], data.y[mask])
        results = metric_results(y_true=data.y[mask].data.cpu().numpy(), y_logits=y_logits[mask].data.cpu().numpy())
        return loss.item(), results

    model = Net(dataset, device, args)
    if args.cuda:
        model = model.to(device)


    if model_pretrain:
        pretrained_dict = model_pretrain.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
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
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(args.epochs)):
        train_loss = train(model, optimizer, dataset, args.batch_train)
        # train_loss, train_res = test(model, train_dataset.data)

        ### inductive
        # val_loss, val_res = test(model, dataset.data, 'val')
        # test_loss, test_res = test(model, dataset.data, 'test')

        ### transductive
        val_loss, val_res = test(model, dataset.data, dataset.data.val_mask)
        test_loss, test_res = test(model, dataset.data, dataset.data.test_mask)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_val_res = val_res
            best_test_res = test_res
            best_epoch = epoch
            # X = model.get_emb(dataset.data.x, dataset.data.edge_index).cpu().detach().numpy()
            # context_gate, feature_gate, topology_gate = model.get_gate(dataset.data.x, dataset.data.edge_index)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
    ############# plot TSNE embedding
    # X_embedded = TSNE(n_components=2, perplexity=100, learning_rate=200, early_exaggeration=1, verbose=1, init='random').fit_transform(X.cpu().detach().numpy())
    # df = pd.DataFrame()
    # df['x'] = X_embedded[:, 0]
    # df['y'] = X_embedded[:, 1]
    # df['class'] = dataset.data.y.cpu().detach().numpy()
    # plt.figure()
    # sns.scatterplot(x='x', y='y', hue='class', alpha=0.5, data=df)
    # plt.savefig('{}_{}_TSNE.jpg'.format(args.net, args.dataset))
    ############## plot distance distribution
    # import pdb;pdb.set_trace()

    # print('Calculating pairwise Cosine Similarity...')
    # start_time = time.time()
    # Y = dataset.data.y.cpu().detach().numpy()
    # X_norm, X_abnorm = X[Y==0, :], X[Y==1, :]
    # if args.dataset == 'YelpChi': # random sample nodes, otherwise out of memory
    #     X_norm = X_norm[np.random.choice(X_norm.shape[0], X_norm.shape[0] // 4, replace=False), :]
    #     X_abnorm = X_abnorm[np.random.choice(X_abnorm.shape[0], X_abnorm.shape[0] // 4, replace=False), :]
    # D_norm = cosine_similarity(X_norm)
    # D_abnorm = cosine_similarity(X_abnorm)
    # D_na = cosine_similarity(X_norm, X_abnorm).flatten()
    # # D_norm = LA.norm(X_norm[:, None, :] - X_norm[None, :, :], axis=-1)
    # # D_abnorm = LA.norm(X_abnorm[:, None, :] - X_abnorm[None, :, :], axis=-1)
    # # D_na = LA.norm(X_norm[:, None, :] - X_abnorm[None, :, :], axis=-1).flatten()
    # # remove self distance
    # D_norm = D_norm[np.where(~np.eye(D_norm.shape[0],dtype=bool))]
    # D_abnorm = D_abnorm[np.where(~np.eye(D_abnorm.shape[0],dtype=bool))]
    # # remove outliers
    # # import pdb;pdb.set_trace()
    # D_norm = remove_outliers(D_norm, 6)
    # D_abnorm = remove_outliers(D_abnorm, 6)
    # D_na = remove_outliers(D_na, 6)
    # data = [D_norm.tolist(), D_abnorm.tolist(), D_na.tolist()]
    # legend_labels = ['N-N', 'AN-AN', 'N-AN']
    # # data = np.array(data)
    # metric_name = 'Cosine similarity'
    # Distance = pd.concat([pd.DataFrame(np.vstack((data[ind], [label]*len(data[ind]))).T, columns=[metric_name, 'type']) for ind, label in enumerate(legend_labels)], ignore_index=True)
    # Distance[metric_name] = pd.to_numeric(Distance[metric_name])
    # print('Saving metrics...')
    # Distance.to_pickle('./{}_{}_cosine_similarity.pkl'.format(args.net, args.dataset)) 
    # # Distance.to_pickle('./{}_{}_cosine_similarity.pkl'.format('ICA', args.dataset))  
    # print("--- %s minutes for calculating similarity" % ((time.time() - start_time)/60))
    
    # gates = {'context':context_gate, 'feature':feature_gate, 'topology':topology_gate}
    # with open('./{}_{}_gates.pkl'.format(args.net, args.dataset), 'wb') as handle:
    #     pickle.dump(gates, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return best_epoch, best_test_res, train_losses, val_losses


def run_our(args, dataset, Net):
    aug1 = Aug.Compose([Aug.EdgeRemoving(pe=0.2), Aug.FeatureMasking(pf=0.1)])
    # aug2 = Aug.Compose([Aug.EdgeRemoving(pe=0.2), Aug.FeatureMasking(pf=0.1)])
    # augmentor = (aug1, aug2)
    data = dataset.data
    if args.net == 'Our-GAD':
        pretrain_model = SchemaCL(dataset, args, device, aug1)
    elif args.net == 'SupCL':
        pretrain_model = SupCL(dataset, args, device, aug1)

    # data = laplace_decomp(data, args.max_freqs)

    if args.cuda:
        pretrain_model = pretrain_model.to(device)
        data = data.to(device)

    # optimizer_train = torch.optim.Adam([{'params':pretrain_model.encoder.to_k.lin1.parameters(),
    #                                      'weight_decay':args.weight_decay, 'lr':args.lr},
    #                                     {'params':pretrain_model.encoder.to_k.lin2.parameters(),
    #                                      'weight_decay':args.weight_decay, 'lr':args.lr},
    #                                     {'params':pretrain_model.encoder.to_k.prop1.parameters(),
    #                                      'weight_decay':0.0, 'lr':args.lr},
    #                                     {'params':pretrain_model.read.parameters(),
    #                                      'weight_decay':args.weight_decay, 'lr':args.lr}], lr=args.lr)
    optimizer_train = torch.optim.Adam(pretrain_model.parameters(), weight_decay=args.weight_decay,lr=args.lr)

    pretrain_losses = []
    for epoch in tqdm(range(args.epochs)):
        ## for mini-batch train
        # train_idx = np.arange(data.num_nodes)[data.train_mask.cpu()]
        # random.shuffle(train_idx)
        # num_batches = int(len(train_idx) / args.batchsize) + 1
        # loss = 0
        # for batch in range(num_batches):
        #     i_start = batch*args.batchsize
        #     i_end = min((batch+1)*args.batchsize, len(train_idx))
        #     batch_idx = train_idx[i_start:i_end]
        #     batch_idx_dict = {batch_idx[i]:i for i in range(len(batch_idx))}
        #     batch_edge_index = []
        #
        #     for v in batch_idx:
        #         temp_idx_list = (data.edge_index == v).nonzero(as_tuple=False)
        #         temp2 = list(data.edge_index.index_select(1, temp_idx_list[:,1]).cpu().numpy())
        #         for i in range(len(temp2[0])):
        #             if temp2[0][i] == v:
        #                 if temp2[1][i] in batch_idx:
        #                     batch_edge_index.append([batch_idx_dict[v], batch_idx_dict[temp2[1][i]]])
        #             elif temp2[0][i] in batch_idx:
        #                 batch_edge_index.append([batch_idx_dict[temp2[0][i]], batch_idx_dict[v]])
        #     batch_edge_index = torch.tensor(np.array(batch_edge_index).T).type(torch.LongTensor)
        #
        #     if args.cuda:
        #         batch_edge_index = batch_edge_index.to(device)
        pretrain_model.train()
        optimizer_train.zero_grad()
        if args.net == 'Our-GAD':
            loss_pretrain = pretrain_model(data.x, data.edge_index)  #
        elif args.net == 'SupCL':
            loss_pretrain = pretrain_model(data.x, data.edge_index, data.y[data.train_mask], data.train_mask)
            # loss_pretrain = pretrain_model(data.x[data.train_mask], data.train_edge_index, data.y[data.train_mask])
        loss_pretrain.backward()
        optimizer_train.step()
        pretrain_losses.append(loss_pretrain.detach().item())
            # loss += loss_pretrain.item()
        #
        # print(f'Epoch:{epoch}, loss:{loss / num_batches}')
    print('Pre-training of SCL is done!')

    # fine_tuning process

    best_epoch, best_test_res, train_losses, val_losses = run(args, dataset, Net, pretrain_model)
    return best_epoch, best_test_res, pretrain_losses, train_losses, val_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.4)
    parser.add_argument('--val_rate', type=float, default=0.1)
    parser.add_argument('--batchsize', type=int, default=1024, help='batch size 1024 for yelp, 256 for amazon.')
    parser.add_argument('--dataset', type=str, choices= ['YelpChi', 'Amazon', 'alpha', 'otc', 'epinions'], default='YelpChi')
    parser.add_argument('--net', type=str, choices=['SupCL', 'CARE-GNN', 'Our-GAD', 'GPR-GNN', 'GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'LambdaNet'], default='Our-GAD')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--RPMAX', type=int, default=5, help='repeat count')
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.0)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', type=str, default=None)
    parser.add_argument('--heads', default=1, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--dim_k', default=16, type=int, help='the dimension of key')
    parser.add_argument('--neigh_num', default=10, type=int)
    parser.add_argument('--resample_count', type=int, default=5)
    parser.add_argument('--max_freqs', type=int, default=20, help='select the max_freqs lowest engenvalues after Laplacian Decomposition')
    parser.add_argument('--batch_train', type=bool, default=False)
    parser.add_argument('--num_hops', default=2, type=int)
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--LPE_dim', default=16, type=int)  #should be the same as dim_k
    parser.add_argument('--LPE_layers', default=1, type=int)
    parser.add_argument('--LPE_heads', default=1, type=int)
    parser.add_argument('--PE_train', default=False, type=bool)
    parser.add_argument('--lambda_encoder', default='GPRGNN', type=str)
    parser.add_argument('--no_cuda', type=int, default=0)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if args.cuda else torch.device("cpu"))
    print(f'run on {args.dataset}')

    dataset = LoadDataSet(args.dataset)
    adj = to_dense_adj(dataset.data.edge_index).squeeze()
    UU,SS,VV = torch.svd_lowrank(adj, q=args.max_freqs)
    dataset.data.a = VV
    data = dataset.data #.to(device)
    # not for otc, alpha dataset
    # print('homophily ratio:', homophily(dataset.data.edge_index, dataset.data.y, method='node'))
    real_mask = (data.train_mask.float() + data.val_mask.float() + data.test_mask.float()).type(torch.bool)
    if args.dataset in ['Amazon', 'YelpChi']:
        print('homophily ratio:', homophily(data.edge_index, data.y, method='node'))
    else:
        print('homophily ratio:', homophily_(data.edge_index, data.y, real_mask))

    print('number of nodes:', data.num_nodes)
    labeled_nodes = sum(data.train_mask.float()) + sum(data.val_mask.float()) + sum(data.test_mask.float())
    print('labeled nodes:', labeled_nodes.item())
    print('number of edge index:', data.edge_index.shape[1]/2)

    print('abnormal rate:', (torch.count_nonzero(data.y[data.train_mask]) + torch.count_nonzero(data.y[data.val_mask]) +
                             torch.count_nonzero(data.y[data.test_mask]))/labeled_nodes)

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
    else:
        Net = Classifier

    res = []
    for RP in range(args.RPMAX):
        print(RP)
        if gnn_name == 'Our-GAD' or gnn_name == 'SupCL':
            best_epoch, best_test_res, pretrain_losses, train_losses, val_losses = run_our(args, dataset, Net)
        else:
            best_epoch, best_test_res, train_losses, val_losses = run(args, dataset, Net)
        print(best_epoch, best_test_res)
        res.append([best_test_res['auc'], best_test_res['ap'], best_test_res['acc'], best_test_res['recall'], best_test_res['F1_macro'], best_test_res['F1_micro']])
        # if pretrain_losses is not None:
        #     plt.figure()
        #     plt.plot(pretrain_losses)
        #     plt.xlabel('epoch')
        #     plt.ylabel('loss')
        #     plt.title('pretrain loss repeat {}'.format(RP))
        #     plt.savefig('{}_pretrain_loss_{}.jpg'.format(args.dataset, RP))
        # plt.figure()
        # plt.plot(train_losses, label='train loss')
        # plt.plot(val_losses, label='val loss')
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.title('finetune loss repeat {}'.format(RP))
        # plt.legend()
        # plt.savefig('{}_finetune_loss_{}.jpg'.format(args.dataset, RP))
    print('average results:', np.mean(res, axis=0) * 100)











