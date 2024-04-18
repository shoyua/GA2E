from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
from scripts.utils import  create_norm
import numpy as np
import pickle, random

import dgl.function as fn
import numpy as np
from scipy.stats import gaussian_kde
import torch.nn.functional as F
import dgl



def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            # num_classes=num_classes,
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class Discrimator(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,                        
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            concat_hidden: bool = False,            
         ):
        super(Discrimator, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate        

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1
        

        # build discrimator
        self.discrimator = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            # num_classes=num_classes,
            in_dim=in_dim,
            num_hidden=enc_num_hidden, 
            out_dim=enc_num_hidden,
            num_layers=num_layers,            
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.norm = nn.BatchNorm1d(in_dim)
        self.linear0 = nn.Linear(num_hidden, num_hidden)
        self.linear = nn.Linear(num_hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU()   
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout(0.5)
        self.tanh = nn.Tanh()


    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    
    def forward(self, graph_list, device):     
        if isinstance(graph_list, list):
            # graph_list=[dgl.to_bidirected(graph.to("cpu"),copy_ndata=True) for graph in graph_list]
            # g=dgl.batch(graph_list).to(next(self.parameters()).device)
            g=dgl.batch(graph_list).to(device)
        else:
            g = graph_list.to(device)
                        
        if self._encoder_type == "mlp":
            fake_out = self.discrimator(g.ndata['attr'])  
        else:            
            fake_out = self.discrimator(g, g.ndata['attr']) 
        g.ndata['h'] = fake_out
        fake_out = dgl.mean_nodes(g, 'h')
        fake_out = self.prelu(fake_out)       
        fake_out = self.linear(fake_out)
        fake_out = self.sigmoid(fake_out)  
        return fake_out
        
