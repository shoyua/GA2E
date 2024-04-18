from ast import Raise
import datetime
import json
import random
import shutil
from scripts.logger import Logger
from scripts.utils import load_dataset, load_graph_data, GRAPH_DICT, GRAPH_LEVEL_DICT, batch_generator, create_optimizer, get_node_subgraphs, get_edge_subgraphs, rec_loss_fun, get_edge_split, get_test_edge_pair
import sys
from tqdm import tqdm
import os
import torch
from torch import nn
import torch.optim as optim
from args import read_args
import dgl
import numpy as np
import torch.nn.functional as F
from models import build_gen_model, build_dis_model
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from ogb.lsc import PCQM4Mv2Evaluator
from sklearn.ensemble import GradientBoostingRegressor
from pprint import pprint
import yaml
torch.set_num_threads(1)


def merge_graph(gen_graphs, ori_graphs, flag='dis'):
    merged_graphs = gen_graphs + ori_graphs
    if flag == 'dis':
        labels = [0] * len(gen_graphs) + [1] * len(ori_graphs)
    elif flag == 'gen':
        labels = [1] * len(gen_graphs) + [1] * len(ori_graphs)
    index_list = list(range(len(merged_graphs)))
    random.shuffle(index_list)
    shuffled_graphs = [merged_graphs[i] for i in index_list]
    shuffled_labels = torch.tensor([labels[i] for i in index_list]).float()
    return shuffled_graphs, shuffled_labels


def get_node_split(graph):
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    train_ids = torch.nonzero(train_mask, as_tuple=True)[0].tolist()
    val_ids = torch.nonzero(val_mask, as_tuple=True)[0].tolist()
    test_ids = torch.nonzero(test_mask, as_tuple=True)[0].tolist()
    node_labels = graph.ndata['label']
    return train_ids, val_ids, test_ids, node_labels


def get_embed(args, model, graph_list):
    '''
    获取模型encoder输出的emb
    '''
    dataloader = list(batch_generator(graph_list, args.batch, False))
    graph_emb = []
    for idx, batched_graph in enumerate(dataloader):
        batched_graph = [dgl.to_bidirected(
            g, copy_ndata=True) for g in batched_graph]
        bg = dgl.batch(batched_graph).to(args.device)
        emb = model.embed(bg, bg.ndata['feat'])
        bg.ndata['h'] = emb
        emb = dgl.mean_nodes(bg, 'h').detach().cpu()
        # emb = dgl.sum_nodes(bg, 'h').detach().cpu()
        graph_emb.append(emb)
    concatenated_tensor = F.normalize(
        torch.cat(graph_emb, dim=0), dim=-1).detach().cpu().numpy()
    return concatenated_tensor


def get_ori_embed(args, model, graph_list):
    dataloader = list(batch_generator(graph_list, args.batch, False))
    graph_emb = []
    for idx, batched_graph in enumerate(dataloader):
        batched_graph = [dgl.to_bidirected(
            g.to("cpu"), copy_ndata=True) for g in batched_graph]
        g = dgl.batch(batched_graph)
        emb = dgl.mean_nodes(g, 'feat')
        # emb = dgl.sum_nodes(g, 'feat')
        graph_emb.append(emb)
    concatenated_tensor = torch.cat(graph_emb, dim=0).detach().cpu().numpy()
    return concatenated_tensor


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool_)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def graph_feat_reshape(graph):
    graph.ndata['feat'] = graph.ndata['feat'].float().reshape(-1, 1)
    return graph


def evaluation(args, model, log, train_graphs, test_graphs, train_ids, test_ids, labels):
    model.eval()
    # if args.task == 'lp':
    #     train_graphs = train_graphs[:10000]
    #     train_ids = train_ids[:10000]
    train_emb = get_embed(args, model, train_graphs)
    test_emb = get_embed(args, model, test_graphs)
    # train_emb = get_ori_embed(args, model, train_graphs)
    # test_emb = get_ori_embed(args, model, test_graphs)
    labels = labels.detach().cpu().numpy()
    if args.task == 'nc':
        clf = LogisticRegression(solver='liblinear')
    elif args.task == 'lp':
        clf = LogisticRegression()
    elif args.task == 'gc':
        clf = LinearRegression()
    if args.task != 'lp':
        clf.fit(train_emb, labels[train_ids])
    if args.task == "nc":
        y_pred = clf.predict_proba(test_emb)
        labels = labels.reshape(-1, 1)
        onehot_encoder = OneHotEncoder(categories='auto').fit(labels)
        labels = onehot_encoder.transform(labels).toarray().astype(np.bool_)
        y_pred = prob_to_one_hot(y_pred)
        micro = f1_score(labels[test_ids], y_pred, average="micro")
        macro = f1_score(labels[test_ids], y_pred, average="macro")
        acc = accuracy_score(labels[test_ids], y_pred)
        res = {
            'F1Mi': micro,
            'F1Ma': macro,
            'acc': acc}
    elif args.task == "lp":
        # y_pred = clf.predict(test_emb)
        # auc_score = roc_auc_score(labels[test_ids], y_pred)
        # ap_score = average_precision_score(labels[test_ids], y_pred)
        product = train_emb*test_emb
        y_pred = np.sum(product, axis=1)
        # def sigmoid(x):
        #     return 1 / (1 + np.exp(-x))
        # y_pred = sigmoid(y_pred)
        auc_score = roc_auc_score(labels[test_ids], y_pred)
        ap_score = average_precision_score(labels[test_ids], y_pred)
        res = {
            'auc': auc_score,
            'ap': ap_score
        }
    elif args.task == "gc":
        y_pred = clf.predict(test_emb)
        evaluator = PCQM4Mv2Evaluator()
        input_dict = {'y_pred': y_pred, 'y_true': labels[test_ids]}
        result_dict = evaluator.eval(input_dict)
        res = result_dict
    log.info(res)
    return res


def train(args, log):
    tmp_save_dir = os.path.join(sys.path[0], 'tmp', f'{args.data}')
    if not os.path.exists(tmp_save_dir):
        os.makedirs(tmp_save_dir)
    if args.data in GRAPH_DICT or args.data.startswith("ogbn"):
        graph, num_class = load_dataset(args.data)
        if args.task == "nc":
            nc_data_file = os.path.join(
                tmp_save_dir, f'{args.task}_pre{args.pre_sample_strategy}_test{args.test_sample_strategy}_subg{args.subgraphs}_rwlen{args.walk_length}_rwpath{args.paths}.pt')
            if os.path.exists(nc_data_file):
                train_ids, test_ids, labels, train_graphs, test_graphs, graph_list = torch.load(
                    nc_data_file)
            else:
                train_ids, val_ids, test_ids, labels = get_node_split(graph)
                train_graphs = get_node_subgraphs(
                    graph, train_ids, args, args.test_sample_strategy)
                test_graphs = get_node_subgraphs(
                    graph, test_ids, args, args.test_sample_strategy)
                graph_list = get_node_subgraphs(
                    graph, args.subgraphs, args, args.pre_sample_strategy)
                torch.save([train_ids, test_ids, labels, train_graphs,
                           test_graphs, graph_list], nc_data_file)
            # print(graph_list)
        elif args.task == "lp":
            lp_data_file = os.path.join(
                tmp_save_dir, f'{args.task}_pre{args.pre_sample_strategy}_test{args.test_sample_strategy}_subg{args.subgraphs}_rwlen{args.walk_length}_rwpath{args.paths}.pt')
            if os.path.exists(lp_data_file):
                train_ids, test_ids, labels, train_graphs, test_graphs, graph_list = torch.load(
                    lp_data_file)
                # graph_list=list(train_graphs)
            else:
                train_ids, test_ids, labels, train_graphs, test_graphs = get_test_edge_pair(
                    graph, args)
                # args.sample_strategy == 'rw'
                graph_list = get_edge_subgraphs(
                    graph, args.subgraphs, args, args.pre_sample_strategy)
                torch.save([train_ids, test_ids, labels, train_graphs,
                           test_graphs, graph_list], lp_data_file)
            # print(graph_list)
    elif args.data in GRAPH_LEVEL_DICT:
        # log.info("load data ing")
        gc_data_file = os.path.join(tmp_save_dir, f'{args.task}.pt')
        if os.path.exists(gc_data_file):
            train_ids, test_ids, labels, train_graphs, test_graphs, graph_list = torch.load(
                gc_data_file)
        else:
            train_set, test_set = load_graph_data(args.data)
            train_graphs, train_labels = zip(*train_set)
            train_graphs = list(train_graphs)
            test_graphs, test_labels = zip(*test_set)
            test_graphs = list(test_graphs)
            if len(train_graphs[0].ndata['feat'].shape) == 1:
                train_graphs = [graph_feat_reshape(
                    graph) for graph in train_graphs]
                test_graphs = [graph_feat_reshape(
                    graph) for graph in test_graphs]
            labels = torch.tensor(list(train_labels)+list(test_labels))
            train_ids, test_ids = list(range(len(train_graphs))), list(
                range(len(train_graphs), len(test_graphs)+len(train_graphs)))
            graph_list = train_graphs
            torch.save([train_ids, test_ids, labels, train_graphs,
                        test_graphs, graph_list], gc_data_file)
    else:
        raise ValueError(f"no such dataset: {args.data}")
    log.info(f'load data: {args.data} {len(graph_list)}')
    dataloader = list(batch_generator(graph_list, args.batch))
    # generator
    try:
        # bug fix 20231107 zinc 1-dim feat
        num_features = graph_list[0].ndata["feat"].shape[1]
    except:
        num_features = 1
    gen_model = build_gen_model(args, num_features).to(args.device)
    gen_opt = create_optimizer(
        args.gen_optimizer, gen_model, args.gen_lr, args.weight_decay)
    # discriminator
    dis_model = build_dis_model(args, num_features).to(args.device)
    dis_opt = create_optimizer(
        args.dis_optimizer, dis_model, args.dis_lr, args.weight_decay)
    for epoch in tqdm(range(args.epochs)):
        gen_model.train()
        dis_model.train()
        ###################### discriminator########################
        for idx, batched_graph in enumerate(dataloader):
            if len(graph_list) > 5000 and (random.random() < 0.8):
                continue
            # print(idx, batched_graph)
            gen_batch_graph, _, _ = gen_model(batched_graph)
            # print(gen_batch_graph)
            # advarsarial loss
            merged_graphs, merged_labels = merge_graph(
                gen_batch_graph, batched_graph)
            dis_logits = dis_model(merged_graphs).squeeze(-1)
            adv_loss_fun = nn.BCELoss(reduction='mean')
            loss_dis = adv_loss_fun(dis_logits, merged_labels.to(args.device))
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()
            # log.info(f"discriminator acc{accuracy_score(merged_labels.detach().cpu().numpy(),(dis_logits>0.5).detach().cpu().numpy())}")
        ###################### generator########################
        for idx, batched_graph in enumerate(dataloader):
            if len(graph_list) > 5000 and (random.random() < 0.8):
                continue
            # print(idx, batched_graph)
            gen_batch_graph, mean, var = gen_model(batched_graph)
            #  reconstruct loss
            rec_loss = rec_loss_fun(
                gen_batch_graph, batched_graph, args.device)
            # advarsarial loss
            merged_graphs, merged_labels = merge_graph(
                gen_batch_graph, [], 'gen')
            dis_logits = dis_model(merged_graphs).squeeze(-1)
            adv_loss_fun = nn.BCELoss(reduction='mean')
            dis_loss = adv_loss_fun(dis_logits, merged_labels.to(args.device))
            loss_gen = args.alpha*dis_loss+rec_loss
            gen_opt.zero_grad()
            loss_gen.backward()
            gen_opt.step()
    ######################## evaluation#############################
    res = evaluation(args, gen_model, log, train_graphs, test_graphs,
                     train_ids, test_ids, labels)
    log.info(f"[final res] {res}")
    return res


def get_result(res_list):
    all_keys = set().union(*res_list)
    mean_std = {}
    for key in all_keys:
        values = [d[key] for d in res_list if key in d]
        mean = np.mean(values)
        std = np.std(values)
        # mean_std[key] = {'mean': mean, 'std': std}
        mean_std[f"{key}_mean"] = mean
        mean_std[f"{key}_std"] = std
    return mean_std


def read_config(args):
    with open(f'config/{args.data}', 'r') as f:
        config = json.load(f)
    if config['data']['value'] != args.data:
        ValueError("config mismatch")
    for key, val in config.items():
        if key in ['data', 'norm', 'task', 'debug', '_wandb', 'device', 'repeats']:
            continue
        setattr(args, key, val['value'])
    return args


def main():
    now = datetime.datetime.now()
    nowtime = now.strftime("%Y_%m_%d_%H_%M_%S")
    args = read_args()
    args.device = f'cuda:{args.device}'
    filename = "debug"+str(args.debug) + '_' + args.data + \
        '_' + args.task+'_' + nowtime + '.log'
    log = Logger(filename, args.data, level='info')
    log = log.logger
    
    args = read_config(args)
    log.info(args)
    log.info("---arguments---")
    for k, v in vars(args).items():
        log.info(k + ':' + str(v))
    res_list = []
    for i in range(args.repeats):
        res = train(args, log)
        res_list.append(res)
    message = get_result(res_list)
    log.info(message)

if __name__ == "__main__":
    main()
