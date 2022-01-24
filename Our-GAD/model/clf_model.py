import torch
import torch.nn as nn
from model.layers import GraphCNN, MLP
import torch.nn.functional as F
import sys
import pickle
from utils import laplace_decomp
from .DCL_model import GraphLambdaLayer
from torch_geometric.utils import to_dense_adj
sys.path.append("models/")

class MLP(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MLP, self).__init__()
        self.norm_layer = nn.BatchNorm1d(dataset.num_features, affine=False)
        self.final_dropout = args.final_dropout
        self.max_freqs = args.max_freqs
        self.linear_prediction_1 = nn.Linear(dataset.num_features, 300)
        self.linear_prediction_2 = nn.Linear(300, 256)
        self.linear_prediction_3 = nn.Linear(256, args.hidden)
        self.dropout = args.dropout

    def forward(self, x, edge_index):
        # adj = to_dense_adj(edge_index).squeeze()
        # UU,SS,VV = torch.svd_lowrank(adj, q=self.max_freqs)
        # x = torch.cat((x, VV), 1)
        x, edge_index = self.norm_layer(x), edge_index
        x = F.elu(F.dropout(self.linear_prediction_1(x), training=self.training))
        x = F.elu(F.dropout(self.linear_prediction_2(x), training=self.training))
        x = F.elu(F.dropout(self.linear_prediction_3(x), training=self.training))
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Classifier(nn.Module):
    def __init__(self, dataset, args):
        super(Classifier, self).__init__()
        out_channels = dataset.num_classes
        device = torch.device("cuda" if args.cuda else torch.device("cpu"))
        # self.gin = GraphCNN(args.num_layers, args.num_mlp_layers, dataset.data.x.size()[1], args.hidden,
        #                     args.neighbor_pooling_type, device)
        self.encoder = GraphLambdaLayer(dim=dataset.num_features, args=args)
        # self.encoder = GraphCNN(input_dim=dataset.num_features, hidden_dim=args.hidden, device=device)
        self.linear_prediction = nn.Linear(args.hidden, out_channels)
        self.final_dropout = args.final_dropout
        # self.MLP = MLP(dataset, args)
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
        self.softmax = torch.nn.Softmax(dim=1)
        self.max_freqs = args.max_freqs
        self.dataset_name = args.dataset
        self.PE_train = args.PE_train
        if self.PE_train:
            self.adj = to_dense_adj(dataset.data.edge_index).squeeze()
        self.gate1 = nn.Linear(args.hidden * 2, args.hidden)
        self.gate2 = nn.Linear(args.hidden * 2, args.hidden)
        self.sigmoid = nn.Sigmoid()

    def m_loss(self):
        return nn.CrossEntropyLoss()

    def to_prob(self, x, edge_index):
        return self.softmax(self.forward(x, edge_index))

    def get_emb(self, x, edge_index):
        h_1 = self.encoder(x, edge_index)
        h_2 = self.MLP(x)
        product = torch.cat((h_1, h_2), 1)
        h_1 = h_1 * self.sigmoid(self.gate1(product))
        h_2 = h_2 * self.sigmoid(self.gate2(product))
        h_1 = h_1 + h_2
        # h_1= torch.cat((h_1, h_2), 1)
        return h_1
    def forward(self, x, edge_index):
        if self.PE_train:
            UU,SS,VV = torch.svd_lowrank(self.adj, q=self.max_freqs)
            EigVals_1, EigVecs_1 = SS, VV
            EigVals_1 = EigVals_1.unsqueeze(0)
            EigVals_1 = EigVals_1.repeat(EigVecs_1.shape[0], 1).unsqueeze(2)
            # try:
            #     with open(str(self.dataset_name) + '_eigvals_test_high.pickle', 'rb') as f:
            #         EigVals = pickle.load(f)
            #     with open(str(self.dataset_name) + '_eigvecs_test_high.pickle', 'rb') as f2:
            #         EigVecs = pickle.load(f2)
            # except:
            #     EigVals, EigVecs = laplace_decomp(x, edge_index, self.max_freqs)
            #     with open(str(self.dataset_name) + '_eigvals_test_high.pickle', 'wb') as f:
            #         pickle.dump(EigVals, f)
            #     with open(str(self.dataset_name) + '_eigvecs_test_high.pickle', 'wb') as f2:
            #         pickle.dump(EigVecs, f2)
            h_1 = self.encoder(x, edge_index, EigVals_1, EigVecs_1)
        else:
            h_1 = self.encoder(x, edge_index)
        # h_1 = self.gin(x, edge_index, classifier=True)
        # h_2 = self.MLP(x, edge_index)
        h_2 = self.MLP(x)
        product = torch.cat((h_1, h_2), 1)
        h_1 = h_1 * self.sigmoid(self.gate1(product))
        h_2 = h_2 * self.sigmoid(self.gate2(product))
        # h_1 = torch.cat((h_1, h_2), 1)
        h_1 = h_1 + h_2
        score_final_layer = F.dropout(self.linear_prediction(h_1), self.final_dropout, training=self.training)
        return score_final_layer

