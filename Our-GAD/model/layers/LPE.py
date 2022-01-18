
import torch
import torch.nn as nn


class NodeLPE(nn.Module):
    def __init__(self, LPE_dim):
        super(NodeLPE, self).__init__()
        self.linear_A = nn.Linear(2, LPE_dim)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=LPE_dim, nhead=LPE_n_heads)
        # self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LPE_layers)

    def forward(self, EigVecs, EigVals):
        PosEnc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).float()
        # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(PosEnc)  # (Num nodes) x (Num Eigenvectors) x 2

        PosEnc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        PosEnc = torch.transpose(PosEnc, 0, 1).float()  # (Num Eigenvectors) x (Num nodes) x 2
        PosEnc = self.linear_A(PosEnc)  # (Num Eigenvectors) x (Num nodes) x PE_dim

        # 1st Transformer: Learned PE
        # PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:, :, 0])
        #
        # remove masked sequences
        PosEnc[torch.transpose(empty_mask, 0, 1)[:, :, 0]] = float('nan')

        # Sum pooling
        PosEnc = torch.nansum(PosEnc, 0, keepdim=False)
        # PosEnc = torch.sum(PosEnc, dim=0, keepdim=False)
        return PosEnc

