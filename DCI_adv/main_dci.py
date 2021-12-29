import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from util import load_data
from models.clf_model import Classifier
from models.dci import DCI
from sklearn.cluster import KMeans

# PGD attack model
class AttackPGD(nn.Module):
    def __init__(self, model, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'

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
            # import pdb;pdb.set_trace()
            grad_feats_adv = torch.autograd.grad(loss, feats_adv)
            feats_adv = feats_adv.detach() + self.step_size * torch.sign(grad_feats_adv[0].detach())
            feats_adv = torch.min(torch.max(feats_adv, feature - self.epsilon), feature + self.epsilon)
            feats_adv = torch.clamp(feats_adv, 0, 1)
        return feats_adv

sig = torch.nn.Sigmoid()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  

def preprocess_neighbors_sumavepool(edge_index, nb_nodes, device):
    adj_idx = edge_index
        
    adj_idx_2 = torch.cat([torch.unsqueeze(adj_idx[1], 0), torch.unsqueeze(adj_idx[0], 0)], 0)
    adj_idx = torch.cat([adj_idx, adj_idx_2], 1)

    self_loop_edge = torch.LongTensor([range(nb_nodes), range(nb_nodes)])
    adj_idx = torch.cat([adj_idx, self_loop_edge], 1)
        
    adj_elem = torch.ones(adj_idx.shape[1])

    adj = torch.sparse.FloatTensor(adj_idx, adj_elem, torch.Size([nb_nodes, nb_nodes]))

    return adj.to(device)

def evaluate(model, test_graph):
    output = model(test_graph[0], test_graph[1])
    pred = sig(output.detach().cpu())
    test_idx = test_graph[3]
    
    labels = test_graph[-1]
    pred = pred[labels[test_idx, 0].astype('int')].numpy()
    target = labels[test_idx, 1]
    
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(target, pred, pos_label=1)
    auc = metrics.auc(false_positive_rate, true_positive_rate)

    return auc

def finetune(args, model_pretrain, device, test_graph, feats_num):
    # initialize the joint model
    model = Classifier(args.num_layers, args.num_mlp_layers, feats_num, args.hidden_dim, args.final_dropout, args.neighbor_pooling_type, device).to(device)
    
    # replace the encoder in joint model with the pre-trained encoder
    pretrained_dict = model_pretrain.state_dict()
    model_dict = model.state_dict()
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion_tune = nn.BCEWithLogitsLoss()

    res = []
    train_idx = test_graph[2]
    node_train = test_graph[-1][train_idx, 0].astype('int')
    label_train = torch.FloatTensor(test_graph[-1][train_idx, 1]).to(device)
    for _ in range(1, args.finetune_epochs+1):
        model.train()
        output = model(test_graph[0], test_graph[1])
        loss = criterion_tune(output[node_train], torch.reshape(label_train, (-1, 1)))
        
        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # testing
        model.eval()
        auc = evaluate(model, test_graph)
        res.append(auc)

    return np.max(res)

def main():
    parser = argparse.ArgumentParser(description='PyTorch deep cluster infomax')
    parser.add_argument('--dataset', type=str, default="wiki",
                        help='name of dataset (default: wiki)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers (default: 2)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='number of hidden units (default: 128)')
    parser.add_argument('--finetune_epochs', type=int, default=100,
                        help='number of finetune epochs (default: 100)')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='number of folds (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num_cluster', type=int, default=2,
                        help='number of clusters (default: 2)')
    parser.add_argument('--recluster_interval', type=int, default=20,
                        help='the interval of reclustering (default: 20)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over neighboring nodes: sum or average')
    parser.add_argument('--training_scheme', type=str, default="decoupled", choices=["decoupled", "joint"],
                        help='Training schemes: decoupled or joint')
    args = parser.parse_args()

    setup_seed(0)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # data loading
    edge_index, feats, split_idx, label, nb_nodes = load_data(args.dataset, args.num_folds)
    input_dim = feats.shape[1]
    # pre-clustering and store userID in each clusters
    kmeans = KMeans(n_clusters=args.num_cluster, random_state=0).fit(feats)
    ss_label = kmeans.labels_
    cluster_info = [list(np.where(ss_label==i)[0]) for i in range(args.num_cluster)]
    # the shuffled features are used to contruct the negative sample-pairs
    idx = np.random.permutation(nb_nodes)
    shuf_feats = feats[idx, :]

    adj = preprocess_neighbors_sumavepool(torch.LongTensor(edge_index), nb_nodes, device)
    feats = torch.FloatTensor(feats).to(device)
    shuf_feats = torch.FloatTensor(shuf_feats).to(device)

    
    model_pretrain = DCI(args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, args.neighbor_pooling_type, device).to(device)
    config = {
        'epsilon': 0.0003,
        'num_steps': 5,
        'step_size': 0.0001,
        'random_start': True,
        'loss_func': 'xent',
    }
    net = AttackPGD(model_pretrain, config)
    # pre-training process with adversarial attack
    if args.training_scheme == 'decoupled':
        optimizer_train = optim.Adam(model_pretrain.parameters(), lr=args.lr)
        for epoch in tqdm(range(1, args.epochs + 1)):
            net.train()
            feats_adv = net(feats, adj, cluster_info, args.num_cluster)
            model_pretrain.train()
            loss_pretrain = model_pretrain(feats, shuf_feats, adj, None, None, None, cluster_info, args.num_cluster)
            loss_pretrain = loss_pretrain + model_pretrain(feats_adv, shuf_feats, adj, None, None, None, cluster_info, args.num_cluster)
            if optimizer_train is not None:
                optimizer_train.zero_grad()
                loss_pretrain.backward()         
                optimizer_train.step()
            # re-clustering
            if epoch % args.recluster_interval == 0 and epoch < args.epochs:
                model_pretrain.eval()
                emb = model_pretrain.get_emb(feats, adj)
                kmeans = KMeans(n_clusters=args.num_cluster, random_state=0).fit(emb.detach().cpu().numpy())
                ss_label = kmeans.labels_
                cluster_info = [list(np.where(ss_label==i)[0]) for i in range(args.num_cluster)]
        
        print('Pre-training Done!')
            
    #fine-tuning process
    fold_idx = 1
    every_fold_auc = []
    for (train_idx, test_idx) in split_idx:
        test_graph = (feats, adj, train_idx, test_idx, label)
        tmp_auc = finetune(args, model_pretrain, device, test_graph, input_dim)
        every_fold_auc.append(tmp_auc)
        print('AUC on the Fold'+str(fold_idx)+': ', tmp_auc)
        fold_idx += 1
    print('The averaged AUC score: ', np.mean(every_fold_auc))


if __name__ == '__main__':
    main()
