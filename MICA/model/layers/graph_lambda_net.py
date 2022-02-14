import torch
from torch import nn, einsum
from torch.nn import Linear
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import GCNConv
from model.layers.LPE import NodeLPE

from model.GNN_models import GPR_prop
# helpers functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos



class GPRGNN_encoder(torch.nn.Module):
    def __init__(self, dim_in, dim_out, args, hidden_dim=64):
        super(GPRGNN_encoder, self).__init__()
        self.lin1 = Linear(dim_in, hidden_dim)
        self.lin2 = Linear(hidden_dim, dim_out)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return x


# lambda layer
class GraphLambdaLayer(nn.Module):
    def __init__(
        self,
        dim,  #channels going in
        args,
        dim_u = 1):
        super().__init__()

        dim_out = args.hidden  # channels out
        # dim_out = default(dim_out, dim)
        dim_k = args.dim_k  #key dimension
        dim_LPE = args.LPE_dim

        self.u = dim_u # intra-depth dimension
        self.heads = args.heads  #number of heads, for multi-query
        assert (dim_out % self.heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // self.heads
        self.lambda_encoder = args.lambda_encoder
        self.to_q = nn.Linear(dim, dim_k * self.heads, bias = False)  #
        if self.lambda_encoder == 'gcn':
            self.to_k = GCNConv(dim, dim_k * dim_u, bias = False)  # or use GPRGNN
            self.to_v = GCNConv(dim, dim_v * dim_u, bias = False)
        elif self.lambda_encoder == 'GPRGNN':
            self.to_k = GPRGNN_encoder(dim_in=dim, dim_out=dim_k*dim_u, args=args)
            self.to_v = GPRGNN_encoder(dim_in=dim, dim_out=dim_v*dim_u, args=args)
        elif self.lambda_encoder == 'MLP':
            self.to_k = nn.Linear(dim, dim_k*dim_u, bias=False)
            self.to_v = nn.Linear(dim, dim_v*dim_u, bias=False)

        self.norm_q = nn.BatchNorm1d(dim_k * self.heads)
        self.norm_v = nn.BatchNorm1d(dim_v * dim_u)

        self.node_pos = NodeLPE(LPE_dim=dim_LPE)
        self.batch_norm = nn.BatchNorm1d(dim, affine=False)

    def forward(self, x, edge_index, EigVals=None, EigVecs=None):
        b, d, u, h = *x.shape, self.u, self.heads

        x = self.batch_norm(x)

        q = self.to_q(x)
        if self.lambda_encoder == 'MLP':
            k = self.to_k(x)
            v = self.to_v(x)
        else:
            k = self.to_k(x, edge_index)
            v = self.to_v(x, edge_index)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b (h k) -> b h k', h = h)
        k = rearrange(k, 'b (u k) -> b u k', u = u)
        v = rearrange(v, 'b (u v) -> b u v', u = u)

        k = k.softmax(dim=-1)

        λc = einsum('b u k, b u v -> b k v', k, v)
        Yc = einsum('b h k, b k v -> b v', q, λc)

        if EigVecs is not None:
            device = x.device
            EigVals = EigVals.to(device)
            EigVecs = EigVecs.to(device)
            self.node_pos = self.node_pos.to(device)

            pos_emb = self.node_pos(EigVecs, EigVals)
            pos_emb = rearrange(pos_emb, 'b (u k) -> b u k', u = u)

            λp = einsum('b u k, b u v -> b k v', pos_emb, v)
            Yp = einsum('b h k, b k v-> b v', q, λp)

            Y = Yc + Yp
        else:
            Y = Yc
        # out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        return Y

if __name__ == '__main__':
    x = torch.randn(1,32,64,64)
    # global context
    layer = GraphLambdaLayer(dim=32, dim_out=32, n=64, dim_k=16, heads=2, dim_u=1)
    out = layer(x)

