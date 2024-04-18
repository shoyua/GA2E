import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from torch.utils.data import Dataset, DataLoader

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.graph_conv = GraphConv(in_feats, out_feats)

    def forward(self, g, inputs):
        return self.graph_conv(g, inputs)

class GraphClassifier(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GraphClassifier, self).__init__()
        self.layers = nn.ModuleList([
            GCNLayer(in_feats, hidden_size),
            GCNLayer(hidden_size, num_classes)
        ])

    def forward(self, g, inputs):
        features = inputs
        for layer in self.layers:
            features = F.relu(layer(g, features))
        g.ndata['features'] = features
        return dgl.mean_nodes(g, 'features')