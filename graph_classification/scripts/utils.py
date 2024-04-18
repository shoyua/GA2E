from functools import partial
import os
import sys
import numpy as np
import torch
import random
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as F
from torch import nn
import dgl
from dgl import AddSelfLoop
from dgl.data import (
    load_data,
    TUDataset,
    CoraGraphDataset,
    CiteseerGraphDataset,
    PubmedGraphDataset,
    CoauthorPhysicsDataset,
    CoauthorCSDataset,
    AmazonCoBuyPhotoDataset,
    
    BA2MotifDataset
)

from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tqdm import tqdm
from torch import optim as optim
from collections import namedtuple, Counter
import wandb, os, yaml

import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,    
    "coacs": CoauthorCSDataset,
    "coaphysics": CoauthorPhysicsDataset,
    "coaphoto": AmazonCoBuyPhotoDataset,
    # "coacomputer": AmazonCoBuyComputerDataset,    
}

GRAPH_LEVEL_DICT = {
    # "zinc": ZINCDataset,
    'BA2M': BA2MotifDataset,
    "imdb-binary" : TUDataset,
    "imdb-multi" : TUDataset,
    "proteins" : TUDataset,
    "collab": TUDataset,
    "mutag": TUDataset,
    "reddit-binary": TUDataset,
    "nci1": TUDataset
}

def preprocess(graph):
    # feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    # graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    # return np.array(mask, dtype=np.bool)
    return torch.tensor(mask, dtype=torch.bool)


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

 


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.data not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.data]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset , (feature_dim, num_classes)
 

def find_khop_neighbors(graph, id, k):
    if k == 0:
        return []
    nids = graph.successors(id).tolist()
    temp = []
    for nid in nids:
        nnids = find_khop_neighbors(graph, nid, k-1)
        temp.extend(nnids)
    nids.extend(temp)
    return nids


def khop_neighbors(graph, nids, k):
    neighbors = []
    for id in tqdm(nids, desc='find khop neighbors...'):
        nei = find_khop_neighbors(graph, id, k)
        neighbors.append(set(nei))
    return neighbors


def batch_generator(data_list, batch_size):
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]


def create_optimizer(opt, model, lr, weight_decay):
    '''
    创建优化器，
    输入：优化器的配置
    输出：一个优化器
    '''
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "rmsprop":
        optimizer = optim.RMSprop(parameters, lr=lr)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    else:
        return nn.Identity


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def node_1hop_subg(graph, node):
    
    neighbors = graph.predecessors(node).tolist(
    ) + [node] + graph.successors(node).tolist()
    
    subgraph = graph.subgraph(neighbors)
    return subgraph


def node_rw_subg(g, start_node):
    '''
    提取随机游走子图
    '''
    walk_length = 10
    restart_prob = 0.05
    paths = 5    
    walks, _ = dgl.sampling.random_walk(
        g,
        torch.tensor([start_node]*paths, dtype=torch.int64),
        length=walk_length,
        restart_prob=restart_prob
    )
    
    walk = torch.flatten(walks)
    selected_nodes = torch.where(walk == -1, torch.tensor(start_node), walk)
    selected_nodes = torch.unique(selected_nodes)
    sub_g = g.subgraph(selected_nodes)
    return sub_g
 
def get_node_subgraphs(graph, k, sample_strategy):
    subgraphs = []
    if isinstance(k, list):
        node_sample = k
    else:        
        num_nodes = graph.num_nodes()
        node_sample = random.sample(range(num_nodes), min(k, num_nodes))
    for node in node_sample:
        if sample_strategy == '1hop':
            subgraph = node_1hop_subg(graph, node)
        elif sample_strategy == 'rw':
            subgraph = node_rw_subg(graph, node)
        subgraphs.append(subgraph)

    return subgraphs


def get_combined_labels(pos_graphs, neg_graphs):    
    graphs = pos_graphs + neg_graphs
    labels = [1] * len(pos_graphs) + [0] * len(neg_graphs)    
    combined = list(zip(graphs, labels))
    random.shuffle(combined)
    graphs, labels = zip(*combined)
    return list(graphs), labels


def get_edge_split(graph):    
    num_edges = graph.num_edges()
    eids = list(range(num_edges))
    random.shuffle(eids)  # 随机打乱列表中的元素

    test_size = int(len(eids) * 0.10)
    val_size = int(len(eids) * 0.05)
    test_eids = eids[:test_size]
    train_eids = eids[test_size+val_size:]    
    test_edges_pair = graph.find_edges(test_eids)
    test_edges = list(
        zip(test_edges_pair[0].tolist(), test_edges_pair[1].tolist()))
    test_pos_graphs = get_edge_subgraphs(graph, test_edges)
    train_edges_pair = graph.find_edges(train_eids)
    train_edges = list(
        zip(train_edges_pair[0].tolist(), train_edges_pair[1].tolist()))
    train_pos_graphs = get_edge_subgraphs(graph, train_edges)
    
    neg_edges_pair = dgl.sampling.global_uniform_negative_sampling(
        graph, num_edges-val_size)
    neg_edges = list(
        zip(neg_edges_pair[0].tolist(), neg_edges_pair[1].tolist()))
    neg_graphs = get_edge_subgraphs(graph, neg_edges)
    test_neg_graphs = neg_graphs[:test_size]
    train_neg_graphs = neg_graphs[test_size:]
    assert len(train_neg_graphs) == len(train_pos_graphs)
    
    train_graphs, train_labels = get_combined_labels(
        train_pos_graphs, train_neg_graphs)
    test_graphs, test_labels = get_combined_labels(
        test_pos_graphs, test_neg_graphs)
    labels = train_labels+test_labels
    labels = torch.tensor(labels)
    train_ids = torch.arange(0, len(train_graphs))
    test_ids = torch.arange(len(train_graphs), len(
        train_graphs)+len(test_graphs))
    return train_ids, test_ids, labels, train_graphs, test_graphs


def edge_1hop_subg(graph, src, dst):    
    src_neighbors = graph.predecessors(
        src).tolist() + [src] + graph.successors(src).tolist()
    dst_neighbors = graph.predecessors(
        dst).tolist() + [dst] + graph.successors(dst).tolist()
    
    neighbors = list(set(src_neighbors + dst_neighbors))    
    subgraph = graph.subgraph(neighbors)
    return subgraph


def edge_rw_subg(g, src, dst):
    '''
    提取随机游走子图
    '''
    walk_length = 10
    restart_prob = 0.05
    paths = 5
    
    walks, _ = dgl.sampling.random_walk(
        g,
        torch.tensor([src, dst]*paths, dtype=torch.int64),
        length=walk_length,
        restart_prob=restart_prob
    )
    
    walk = torch.flatten(walks)
    selected_nodes = torch.where(walk == -1, torch.tensor(src), walk)
    selected_nodes = torch.unique(selected_nodes)
    sub_g = g.subgraph(selected_nodes)
    return sub_g
 

def get_edge_subgraphs(graph, k, sample_strategy='1hop'):
    subgraphs = []
    if isinstance(k, list):
        edge_sample = k
    else:        
        num_edges = graph.num_edges()
        eids = random.sample(range(num_edges), min(k, num_edges))
        edges = graph.find_edges(eids)
        edge_sample = zip(edges[0].tolist(), edges[1].tolist())
    for src, dst in edge_sample:
        if sample_strategy == '1hop':
            subgraph = edge_1hop_subg(graph, src, dst)
        elif sample_strategy == 'rw':
            subgraph = edge_rw_subg(graph, src, dst)
        subgraphs.append(subgraph)

    return subgraphs


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
 
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def rec_loss_fun(gen_graph, ori_graph, masked_nodes, device):  
    rec_feat = gen_graph.ndata['attr'].to(device)
    ori_feat = ori_graph.ndata['attr'].to(device)
    loss=sce_loss(rec_feat[masked_nodes], ori_feat[masked_nodes])    
    return loss
