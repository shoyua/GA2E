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
from graphmae.utils import  create_norm, info_nce_loss_torch, grace_loss
import numpy as np
import pickle, random
import dgl.function as fn
import numpy as np
from scipy.stats import gaussian_kde
import torch.nn.functional as F
def setup_module(m_type, enc_dec, num_classes, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            num_classes=num_classes,
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
class PreModel(nn.Module):
    def __init__(
            self, in_dim, num_hidden, num_layers, num_classes, nhead, nhead_out,
            activation, feat_drop, attn_drop, negative_slope, residual, up_bound, 
            lower_bound, ratio_split,  norm, mask_rate, 
            encoder_type, decoder_type, mean_encoder_type, var_encoder_type, 
            loss_fn, drop_edge_rate, replace_rate,
            alpha_l, concat_hidden,):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.up_bound = up_bound
        self.lower_bound = lower_bound
        self.ratio_split = ratio_split
        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1
        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 
        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            num_classes=num_classes,
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
        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            num_classes=num_classes,
            in_dim=dec_in_dim,
            # in_dim=dec_num_hidden,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,            
            concat_out=True,
        )
        self.mean_encoder = setup_module(
            m_type=mean_encoder_type,
            enc_dec="decoding",
            num_classes=num_classes,
            in_dim=dec_in_dim,
            num_hidden=dec_in_dim,
            # num_hidden=dec_num_hidden,
            out_dim=dec_in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,            
            concat_out=True,
        )
        self.var_encoder = setup_module(
            m_type=var_encoder_type,
            enc_dec="decoding",
            num_classes=num_classes,
            in_dim=dec_in_dim,                        
            num_hidden=dec_in_dim,
            # num_hidden=dec_num_hidden,
            out_dim=dec_in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,            
            concat_out=True,
        )
        # build discrimator
        # self.discrimator = setup_module(
        #     m_type=encoder_type,
        #     enc_dec="encoding",
        #     in_dim=in_dim,
        #     num_hidden=enc_num_hidden,
        #     out_dim=enc_num_hidden,
        #     num_layers=num_layers,
        #     nhead=enc_nhead,
        #     nhead_out=enc_nhead,
        #     concat_out=True,
        #     activation=activation,
        #     dropout=feat_drop,
        #     attn_drop=attn_drop,
        #     negative_slope=negative_slope,
        #     residual=residual,
        #     norm=norm,
        # )
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.encoder_linear = nn.Linear(dec_in_dim, dec_num_hidden, bias=False)
        self.decoder_linear = nn.Linear(in_dim, dec_num_hidden, bias=False)
        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.advar_loss = nn.CrossEntropyLoss()
        self.linear = nn.Linear(num_hidden, 1)
        self.linear2 = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.classifier_linear = nn.Linear(num_hidden, num_classes) 
    #     for m in self.modules():
    #         self.weights_init(m)
    # def weights_init(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight.data)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0.0)
    @property
    def output_hidden_dim(self):
        return self._output_hidden_size
    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    # def encoding_mask_noise(self, g, x, up_bound, lower_bound, epoch,ratio_split,  mask_rate=0.3):
    #     num_nodes = g.num_nodes()
    #     perm = torch.randperm(num_nodes, device=x.device)
    #     num_mask_nodes = int(mask_rate * num_nodes)
    #     # 1. mask accrording to the degree
    #     # mask_nodes, keep_nodes = self.mask_by_degree(g, mask_rate, up_bound, lower_bound)
    #     #2. mask by pr score      
    #     # path = "/nlp_group/huyulan/code/.jupyter/graph_learning/GraphMAE/nodes_mask_" + str(mask_rate)+".pkl"  
    #     # import os
    #     # if os.path.exists(path):
    #     #     mask_nodes, keep_nodes = pickle.load(open(path, 'rb'))            
    #     #     print("loading successful")
    #     # else:
    #     #     mask_nodes, keep_nodes = self.mask_by_pr(g, mask_rate)
    #     #     pickle.dump([mask_nodes, keep_nodes], open(path, 'wb'))            
    #     # print(mask_nodes.shape, keep_nodes.shape)
    #     # mask_nodes, keep_nodes = self.mask_by_pr(g, mask_rate)        
    #     # 3. mask by weight
    #     # mask_nodes, keep_nodes = self.mask_by_pr_score(g, mask_rate, ratio_split)
    #     # mask_nodes, keep_nodes = self.mask_by_weight(g, mask_rate, epoch)
    #     # 4. random masking        
    #     mask_nodes = perm[: num_mask_nodes]
    #     keep_nodes = perm[num_mask_nodes: ]
    #     # if self._replace_rate > 0:
    #     #     num_noise_nodes = int(self._replace_rate * num_mask_nodes)
    #     #     perm_mask = torch.randperm(num_mask_nodes, device=x.device)
    #     #     token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
    #     #     noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
    #     #     noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
    #     #     out_x = x.clone()
    #     #     out_x[token_nodes] = 0.0
    #     #     out_x[noise_nodes] = x[noise_to_be_chosen]
    #     # else:
    #     out_x = x.clone()
    #     token_nodes = mask_nodes
    #     out_x[mask_nodes] = 0.0
    #     out_x[token_nodes] += self.enc_mask_token
    #     use_g = g.clone()
    #     return use_g, out_x, (mask_nodes, keep_nodes)
    def encoding_mask_noise(self, g, x,mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]
        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        return use_g, out_x, (mask_nodes, keep_nodes)
    def mask_attr_prediction(self, g, x, up_bound, lower_bound,ratio_split,  epoch):#, num, y_onehot):
        # x_new = self.sigmoid(self.linear2(x))
        # pre_use_g, use_x1, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, up_bound, lower_bound, epoch, ratio_split, self._mask_rate)
        pre_use_g, use_x1, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        # pre_use_g, use_x2, (mask_nodes2, keep_nodes) = self.encoding_mask_noise(g, x, up_bound, lower_bound, epoch, ratio_split, self._mask_rate)
        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g
        enc_rep, all_hidden1 = self.encoder(use_g, use_x1, return_hidden=True)
        # enc_rep, all_hidden1 = self.encoder(use_g, use_x1, y_onehot, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden1, dim=1)
        # mean = self.mean_encoder(use_g, enc_rep, y_onehot)
        # var = self.var_encoder(use_g, enc_rep, y_onehot)
        mean = self.mean_encoder(enc_rep)
        var = self.var_encoder(enc_rep)
        mean = F.normalize(mean, dim=-1)
        var = F.normalize(var, dim=-1)
        # noise = torch.randn(enc_rep.size(0), enc_rep.size(1)).to(g.device)
        # z = mean + noise * torch.exp(var).to(g.device)
        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)   
        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0           
        if self._decoder_type == "mlp":            
            recon = self.decoder(rep)
        else:
            recon = self.decoder(use_g, rep)        
        # fake_out = self.discrimator(pre_use_g, recon)
        # mask_real,mask_fake = enc_rep[mask_nodes], fake_out[mask_nodes]                 
        # advar_loss = self.advar_loss(mask_real,mask_fake)
        x_init = x[mask_nodes]     
        x_rec_0 = recon[mask_nodes]
        # enc_rep = F.normalize(enc_rep, dim=-1)
        # recon = F.normalize(recon, dim=-1)
        # enc_rep_1 = self.sigmoid(self.linear(enc_rep))
        # enc_rep_2 = self.classifier_linear(enc_rep)
        # recon_1 = self.sigmoid(self.linear2(recon))
        # return mask_nodes, recon,  x # ,recon_1, x_new, enc_rep, enc_rep_1, enc_rep_2, mean, var
        rec_loss = self.criterion(x_rec_0, x_init)
        return rec_loss
    def negative_samples(self, x, num):
        negs = []
        for _ in range(num):
            negs.append(F.dropout(x, 0.1))
        return negs
    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep
    @property
    def enc_params(self):
        return self.encoder.parameters()
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    def forward(self, g, x, epoch):
        # ---- attribute reconstruction ----
        # loss = self.mask_attr_prediction(g, x, self.up_bound, self.lower_bound, self.ratio_split, epoch, num)
        # # loss_item = {"loss": loss.item()}
        # return loss
        reps = self.mask_attr_prediction(g, x, self.up_bound, self.lower_bound, self.ratio_split, epoch)#, num, y_onehot)        
        return reps
