import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import LoadDataSet, metric_results, preprocess_neighbors_sumavepool
from model.GNN_models import *

from sklearn.cluster import KMeans # for clustering

import TableIt # for beatiful tables output

def run(args, dataset, Net):
    print('Pretraining...')
    # fragment for DCI and DGI (can be moved to another function/place if needed)
    if args.net in ['DCI', 'DGI', 'DGI_MLP']:
        data = dataset.data.cpu()
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
    
            if args.training_scheme == 'decoupled':
                optimizer_train = torch.optim.Adam(model_pretrain.parameters(), lr=args.lr)
                for epoch in tqdm(range(1, args.epochs + 1)):
                    model_pretrain.train()
                    loss_pretrain = model_pretrain(feats, shuf_feats, adj, None, None, None, cluster_info, args.num_cluster)
                    if optimizer_train is not None:
                        optimizer_train.zero_grad()
                        loss_pretrain.backward()         
                        optimizer_train.step()
                    # re-clustering
                    if epoch % args.recluster_interval == 0 and epoch < args.epochs:
                        model_pretrain.eval()
                        emb = model_pretrain.get_emb(data)
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
        train(model, optimizer, dataset, False)
        # train_loss, train_res = test(model, train_dataset.data)
        val_loss, val_res = test(model, dataset.data, dataset.data.val_mask)
        test_loss, test_res = test(model, dataset.data, dataset.data.test_mask)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_res = val_res
            best_test_res = test_res
            best_epoch = epoch
    return best_epoch, best_test_res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--train_rate', type=float, default=0.4)
    parser.add_argument('--val_rate', type=float, default=0.1)
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

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda:" + str(args.device)) if args.cuda else torch.device("cpu") # this is more flexible
    
    print(f'run on {args.dataset}')

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

        
    res = []
    for RP in range(args.RPMAX):
        print('Repeat {}'.format(RP))
        best_epoch, best_test_res = run(args, dataset, Net)
        print(best_epoch, best_test_res)
        res.append([best_test_res['auc'], best_test_res['ap'], best_test_res['acc'], best_test_res['recall'], best_test_res['F1_macro'], best_test_res['F1_micro']])
    avg_res = np.mean(res, axis=0) * 100
    print('average results:', avg_res)
    
    table = [
            [f'epochs = {args.epochs}', f'lr = {args.lr}', f'wd = {args.weight_decay}'],
            [f'early_stop = {args.early_stopping}',f'hidden = {args.hidden}',f'dropout = {args.dropout}'],
            [f'train_rate = {args.train_rate}',f'val_rate = {args.val_rate}',f'batchsize = {args.batchsize}'],
            [f'dataset = {args.dataset}',f'net = {args.net}',f'no_cuda = {args.no_cuda}'],
            [f'RPMAX = {args.RPMAX}',f'K = {args.K}',f'alpha = {args.alpha}'],
            [f'dprate = {args.dprate}',f'Init = {args.Init}',f'Gamma = {args.Gamma}'],
            [f'heads = {args.heads}',f'output_heads = {args.output_heads}',f'final_dropout_DCI_DGI = {args.final_dropout}'],
            [f'device = {args.device}',f'num_cluster_DCI = {args.num_cluster}',f'num_layers_DCI_DGI = {args.num_layers}'],
            [f'num_mlp_layers_DCI_DGI = {args.num_mlp_layers}',f'hidden_dim_DCI_DGI = {args.hidden_dim}',f'neighbor_pooling_type_DCI_DGI = {args.neighbor_pooling_type}'],
            [f'training_scheme_DCI_DGI = {args.training_scheme}',f'recluster_interval_DCI = {args.recluster_interval}',f''],
            [f'auc: {avg_res[0]:.4f}',f'ap: {avg_res[1]:.4f}',f'acc: {avg_res[2]:.4f}'],
            [f'recall: {avg_res[3]:.4f}',f'F1_macro: {avg_res[4]:.4f}',f'F1_micro: {avg_res[5]:.4f}']
            ]
    
    out = open("Results.txt", "a")
    TableIt.printTable(table, out)
    out.close()

