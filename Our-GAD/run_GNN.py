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

device = torch.device('cuda')

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
        return loss, results

    model = Net(dataset, args)
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
    for epoch in tqdm(range(args.epochs)):
        train(model, optimizer, dataset, args.batch_train)
        # train_loss, train_res = test(model, train_dataset.data)

        ### inductive
        # val_loss, val_res = test(model, dataset.data, 'val')
        # test_loss, test_res = test(model, dataset.data, 'test')

        ### transductive
        val_loss, val_res = test(model, dataset.data, dataset.data.val_mask)
        test_loss, test_res = test(model, dataset.data, dataset.data.test_mask)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_res = val_res
            best_test_res = test_res
            best_epoch = epoch
    return best_epoch, best_test_res


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
            # loss += loss_pretrain.item()
        #
        # print(f'Epoch:{epoch}, loss:{loss / num_batches}')
    print('Pre-training of SCL is done!')

    # fine_tuning process

    best_epoch, best_test_res = run(args, dataset, Net, pretrain_model)
    return best_epoch, best_test_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
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
    parser.add_argument('--final_dropout', type=float, default=0.1,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--LPE_dim', default=16, type=int)  #should be the same as dim_k
    parser.add_argument('--LPE_layers', default=1, type=int)
    parser.add_argument('--LPE_heads', default=1, type=int)
    parser.add_argument('--PE_train', default=False, type=bool)
    parser.add_argument('--lambda_encoder', default='GPRGNN', type=str)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda:" + str(args.device) if args.cuda else torch.device("cpu"))
    print(f'run on {args.dataset}')

    dataset = LoadDataSet(args.dataset)
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
        if gnn_name == 'Our-GAD' or gnn_name == 'SupCL':
            best_epoch, best_test_res = run_our(args, dataset, Net)
        else:
            best_epoch, best_test_res = run(args, dataset, Net)
        print(best_epoch, best_test_res)
        res.append([best_test_res['auc'], best_test_res['ap'], best_test_res['acc'], best_test_res['recall'], best_test_res['F1_macro'], best_test_res['F1_micro']])
    print('average results:', np.mean(res, axis=0) * 100)










