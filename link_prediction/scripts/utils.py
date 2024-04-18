from functools import partial
import os
import sys
import numpy as np
import torch
import random
from scipy.spatial.distance import pdist, squareform
import math
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
    AmazonCoBuyComputerDataset,
    ActorDataset,
    ZINCDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tqdm import tqdm
from torch import optim as optim
GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn": DglNodePropPredDataset,
    "coacs": CoauthorCSDataset,
    "coaphysics": CoauthorPhysicsDataset,
    "coaphoto": AmazonCoBuyPhotoDataset,
    "coacomputer": AmazonCoBuyComputerDataset,
    "actor": ActorDataset
}
GRAPH_LEVEL_DICT = {
    "zinc": ZINCDataset
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
def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT or dataset_name.startswith(
        "ogbn"), f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name[:4]](
            dataset_name, root='data/')
    else:
        dataset = GRAPH_DICT[dataset_name](transform=AddSelfLoop(
        ), raw_dir='data/')
        # dataset = GRAPH_DICT[dataset_name](raw_dir='./data')
    # if dataset_name == "ogbn-arxiv":
    if dataset_name.startswith("ogbn"):
        graph, labels = dataset[0]
        if not graph.is_homogeneous:
            graph = dgl.to_homogeneous(graph)
        num_nodes = graph.num_nodes()
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)
        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)
        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat
        train_mask = torch.full(
            (num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full(
            (num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full(
            (num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = preprocess(graph)  
        if dataset_name.startswith("coa"):
            print("------dataset------", dataset_name)
            size = graph.ndata['feat'].shape[0]
            labels = graph.ndata['label']
            indices = torch.arange(len(labels))
            train_ratio = 0.1
            val_ratio = 0.1
            test_ratio = 0.8
            N = graph.number_of_nodes()
            train_num = int(N * train_ratio)
            val_num = int(N * (train_ratio + val_ratio))
            idx = np.arange(N)
            np.random.shuffle(idx)
            train_idx = idx[:train_num]
            val_idx = idx[train_num:val_num]
            test_idx = idx[val_num:]
            train_mask = sample_mask(train_idx, N)
            val_mask = sample_mask(val_idx, N)
            test_mask = sample_mask(test_idx, N)
            graph.ndata['train_mask'] = train_mask
            graph.ndata['val_mask'] = val_mask
            graph.ndata['test_mask'] = test_mask
        elif dataset_name == "actor":
            graph.ndata['train_mask'] = graph.ndata['train_mask'][:, 0]
            graph.ndata['val_mask'] = graph.ndata['val_mask'][:, 0]
            graph.ndata['test_mask'] = graph.ndata['test_mask'][:, 0]
        # graph = graph.remove_self_loop()
        # graph = graph.add_self_loop()
    feat = graph.ndata["feat"]
    num_classes = dataset.num_classes
    return graph, num_classes
def load_graph_data(dataset_name):
    assert dataset_name in GRAPH_LEVEL_DICT, f"Unknow dataset: {dataset_name}."
    training_set = GRAPH_LEVEL_DICT[dataset_name](transform=AddSelfLoop(
    ), raw_dir='data/', mode="train")
    test_set = GRAPH_LEVEL_DICT[dataset_name](transform=AddSelfLoop(
    ), raw_dir='data/', mode="test")
    return training_set, test_set
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
def batch_generator(data_list, batch_size, shuffle=True):
    if shuffle:
        random.shuffle(data_list)
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]
def create_optimizer(opt, model, lr, weight_decay):
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
    neighbors =[node] + graph.predecessors(node).tolist(
    ) +  graph.successors(node).tolist()
    subgraph = graph.subgraph(neighbors[:30])
    return subgraph
def node_rw_subg(g, start_node, args):
    walk_length = args.walk_length
    restart_prob = 0.05
    paths = args.paths
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
def get_node_subgraphs(graph, k, args, sample_strategy):
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
            subgraph = node_rw_subg(graph, node, args)
        subgraphs.append(subgraph)
    return subgraphs
def get_combined_labels(pos_graphs, neg_graphs):
    graphs = pos_graphs + neg_graphs
    labels = [1] * len(pos_graphs) + [0] * len(neg_graphs)
    combined = list(zip(graphs, labels))
    random.shuffle(combined)
    graphs, labels = zip(*combined)
    return list(graphs), labels
def get_test_edge_pair(graph, args):
    num_edges = graph.num_edges()
    eids = list(range(num_edges))
    random.shuffle(eids)  
    test_size = int(len(eids) * 0.10)
    test_eids = eids[:test_size]
    test_edges_pair = graph.find_edges(test_eids)
    test_edges_src, test_edges_des = test_edges_pair[0].tolist(
    ), test_edges_pair[1].tolist()
    test_pos_graphs_src = get_node_subgraphs(
        graph, test_edges_src, args, args.test_sample_strategy)
    test_pos_graphs_des = get_node_subgraphs(
        graph, test_edges_des, args, args.test_sample_strategy)
    test_pos_graphs = list(zip(test_pos_graphs_src, test_pos_graphs_des))
    neg_edges_pair = dgl.sampling.global_uniform_negative_sampling(
        graph, len(test_edges_src))
    neg_edges_src, neg_edges_des = neg_edges_pair[0].tolist(
    ), neg_edges_pair[1].tolist()
    test_neg_graphs_src = get_node_subgraphs(
        graph, neg_edges_src, args,args.test_sample_strategy)
    test_neg_graphs_des = get_node_subgraphs(
        graph, neg_edges_des, args,args.test_sample_strategy)
    test_neg_graphs = list(zip(test_neg_graphs_src, test_neg_graphs_des))
    assert len(test_pos_graphs) == len(test_neg_graphs)
    test_graphs, test_labels = get_combined_labels(
        test_pos_graphs, test_neg_graphs)
    labels = test_labels
    labels = torch.tensor(labels)
    train_ids = torch.arange(0, len(labels))  
    test_ids = torch.arange(0, len(labels))  
    test_graphs_src, test_graphs_des = zip(*test_graphs)
    return (train_ids), test_ids, labels, list(test_graphs_src), list(test_graphs_des)
def get_test_edge_id(graph, args):
    num_edges = graph.num_edges()
    eids = list(range(num_edges))
    random.shuffle(eids)  
    test_size = int(len(eids) * 0.10)
    test_eids = eids[:test_size]
    test_edges_pair = graph.find_edges(test_eids)
    test_edges_src, test_edges_des = test_edges_pair[0].tolist(
    ), test_edges_pair[1].tolist()
    test_pos_ids = list(zip(test_edges_src, test_edges_des))
    neg_edges_pair = dgl.sampling.global_uniform_negative_sampling(
        graph, len(test_edges_src))
    neg_edges_src, neg_edges_des = neg_edges_pair[0].tolist(
    ), neg_edges_pair[1].tolist()
    test_neg_ids = list(zip(neg_edges_src, neg_edges_des))
    test_graphs, test_labels = get_combined_labels(
        test_pos_ids, test_neg_ids)
    labels = test_labels
    labels = torch.tensor(labels)
    train_ids = torch.arange(0, len(labels))  
    test_ids = torch.arange(0, len(labels))  
    test_src_ids, test_des_ids = zip(*test_graphs)
    return (train_ids), test_ids, labels, list(test_src_ids), list(test_des_ids)
def get_edge_split(graph, args):
    num_edges = graph.num_edges()
    eids = list(range(num_edges))
    random.shuffle(eids)  
    test_size = int(len(eids) * 0.10)
    val_size = int(len(eids) * 0.05)
    test_eids = eids[:test_size]
    train_eids = eids[test_size+val_size:][:20000]  
    test_edges_pair = graph.find_edges(test_eids)
    test_edges = list(
        zip(test_edges_pair[0].tolist(), test_edges_pair[1].tolist()))
    # args.sample_strategy = '1hop'
    test_pos_graphs = get_edge_subgraphs(
        graph, test_edges, args, args.test_sample_strategy)
    train_edges_pair = graph.find_edges(train_eids)
    train_edges = list(
        zip(train_edges_pair[0].tolist(), train_edges_pair[1].tolist()))
    train_pos_graphs = get_edge_subgraphs(
        graph, train_edges, args, args.test_sample_strategy)
    neg_edges_pair = dgl.sampling.global_uniform_negative_sampling(
        graph, len(train_edges)+len(test_edges))  
    neg_edges = list(
        zip(neg_edges_pair[0].tolist(), neg_edges_pair[1].tolist()))
    neg_graphs = get_edge_subgraphs(
        graph, neg_edges, args, args.test_sample_strategy)
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
    src_neighbors = [src] + graph.predecessors(
        src).tolist() + graph.successors(src).tolist()
    dst_neighbors = [dst] + graph.predecessors(
        dst).tolist() + graph.successors(dst).tolist()
    neighbors = list(set(src_neighbors[:15] + dst_neighbors[:15]))  
    subgraph = graph.subgraph(neighbors)
    return subgraph
def edge_rw_subg(g, src, dst, args):
    walk_length = args.walk_length
    restart_prob = 0.05
    paths = args.paths
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
def get_edge_subgraphs(graph, k, args, sample_strategy):
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
            subgraph = edge_rw_subg(graph, src, dst, args)
        subgraphs.append(subgraph)
    return subgraphs
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss
def rec_loss_fun(gen_graph, ori_graph, device):
    rec_feat = dgl.batch(gen_graph).ndata['feat'].to(device)
    ori_feat = dgl.batch(ori_graph).ndata['feat'].to(device)
    loss = sce_loss(rec_feat, ori_feat)
    # loss=nn.MSELoss()(rec_feat, ori_feat)
    return loss
