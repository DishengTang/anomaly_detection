import torch
import torch.nn as nn
from model.layers import GraphCNN, MLP
import torch.nn.functional as F
import sys
import pickle
from utils import laplace_decomp
from .DCL_model import GraphLambdaLayer
sys.path.append("models/")


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

        self.softmax = torch.nn.Softmax(dim=1)
        self.max_freqs = args.max_freqs
        self.dataset_name = args.dataset
        self.PE_train = args.PE_train

    def m_loss(self):
        return nn.CrossEntropyLoss()

    def to_prob(self, x, edge_index):
        return self.softmax(self.forward(x, edge_index))

    def forward(self, x, edge_index):
        if self.PE_train:
            try:
                with open(str(self.dataset_name) + '_eigvals_test_high.pickle', 'rb') as f:
                    EigVals = pickle.load(f)
                with open(str(self.dataset_name) + '_eigvecs_test_high.pickle', 'rb') as f2:
                    EigVecs = pickle.load(f2)
            except:
                EigVals, EigVecs = laplace_decomp(x, edge_index, self.max_freqs)
                with open(str(self.dataset_name) + '_eigvals_test_high.pickle', 'wb') as f:
                    pickle.dump(EigVals, f)
                with open(str(self.dataset_name) + '_eigvecs_test_high.pickle', 'wb') as f2:
                    pickle.dump(EigVecs, f2)
            h_1 = self.encoder(x, edge_index, EigVals, EigVecs)
        else:
            h_1 = self.encoder(x, edge_index)
        # h_1 = self.gin(x, edge_index, classifier=True)
        score_final_layer = F.dropout(self.linear_prediction(h_1), self.final_dropout, training=self.training)
        return score_final_layer

