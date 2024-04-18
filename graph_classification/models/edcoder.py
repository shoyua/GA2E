from itertools import chain
import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
from scripts.utils import create_norm, drop_edge
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


class PreModel(nn.Module):
    def __init__(
            self, in_dim, num_hidden, num_layers, nhead, nhead_out,
            activation, feat_drop, attn_drop, negative_slope, residual, norm, mask_rate,
            encoder_type, decoder_type, mean_encoder_type, var_encoder_type, drop_edge_rate, replace_rate,
            concat_hidden,):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
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

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in (
            "gat", "dotgat") else num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
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
            enc_dec="decoding", #维度没对齐
            # enc_dec="encoding",
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
            concat_out=concat_hidden, #维度没对齐
        )

        self.mean_encoder = setup_module(
            m_type=mean_encoder_type,
            enc_dec="decoding",
            # num_classes=num_classes,
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
            # num_classes=num_classes,
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

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(
                dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(
                dec_in_dim, dec_in_dim, bias=False)

        self.encoder_linear = nn.Linear(dec_in_dim, dec_num_hidden, bias=False)
        self.decoder_linear = nn.Linear(in_dim, dec_num_hidden, bias=False)
        # * setup loss function
        self.advar_loss = nn.CrossEntropyLoss()

        self.linear = nn.Linear(num_hidden, 1)
        self.linear2 = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()
 

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size
 
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(
                self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(
                self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[
                :num_noise_nodes]

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

    def mask_attr_prediction(self, g):  # , num, y_onehot):
        x = g.ndata["attr"]
        # x_new = self.sigmoid(self.linear2(x))
        pre_use_g, use_x1, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(
                pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden1 = self.encoder(use_g, use_x1, return_hidden=True)        
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden1, dim=1)
        # for KL loss
        mean = self.mean_encoder(enc_rep)
        var = self.var_encoder(enc_rep)
        mean = F.normalize(mean, dim=-1)
        var = F.normalize(var, dim=-1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type == "mlp":
            recon = self.decoder(rep)
        else:
            recon = self.decoder(use_g, rep)

        g.ndata["attr"] = recon
        # recovered_graph_list = dgl.unbatch(g)
        return g, mask_nodes, mean, var

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

    def forward(self, graph_list, device):          
        if isinstance(graph_list, list):
            g=dgl.batch(graph_list).to(device)
        else:
            g = graph_list.to(device)
        reps = self.mask_attr_prediction(g)
        return reps
