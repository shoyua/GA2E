import datetime
from scripts.logger import Logger
from scripts.utils import *
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
from sklearn.preprocessing import normalize, OneHotEncoder
# from ogb.lsc import PCQM4Mv2Evaluator
from pprint import pprint
from collections import namedtuple, Counter
from torch.utils.data import DataLoader, Dataset
from scripts.eval import *

 
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)




class CustomDataset(Dataset):
    def __init__(self, graph_list, labels=None):
        self.graph_list = graph_list                

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):        
        return self.graph_list[idx]
 

def collate_gc(samples):
    samples =  [dgl.add_self_loop(g[0]) for g in samples]    
    batched_graph = dgl.batch(samples)    
    return batched_graph


def collate_fn(batch):    
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


 

def get_embed(args, model, graph_list, device):    
    dataloader = graph_list
    graph_emb = []
    for idx, batched_graph in enumerate(dataloader):        
        bg = batched_graph.to(device)
        emb = model.embed(bg, bg.ndata['feat'])
        bg.ndata['h'] = emb
        emb = dgl.mean_nodes(bg, 'h')        
        graph_emb.append(emb)
    concatenated_tensor = F.normalize(torch.cat(graph_emb, dim=0), dim=-1).detach().cpu().numpy()
    return concatenated_tensor


def get_ori_embed(args, model, graph_list):    
    dataloader = list(batch_generator(graph_list, args.batch))
    graph_emb = []
    for idx, batched_graph in enumerate(dataloader):
        batched_graph = [dgl.to_bidirected(
            g.to("cpu"), copy_ndata=True) for g in batched_graph]
        g = dgl.batch(batched_graph)
        emb = dgl.mean_nodes(g, 'feat') 
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


def evaluation(args, model, log, train_graphs, test_graphs, train_ids, test_ids, labels, device):
    model.eval()
    if args.task == 'lp':
        train_graphs = train_graphs[:10000]
        train_ids = train_ids[:10000]
    train_emb = get_embed(args, model, train_graphs, device)
    test_emb = get_embed(args, model, test_graphs, device)    
    labels = labels.detach().cpu().numpy()    
    if args.task == 'nc':
        clf = LogisticRegression(solver='liblinear')        
    elif args.task == 'lp':
        clf = LogisticRegression()
    elif args.task == 'gc':        
        clf = LogisticRegression() 

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
        y_pred = clf.predict(test_emb)
        auc_score = roc_auc_score(labels[test_ids], y_pred)
        ap_score = average_precision_score(labels[test_ids], y_pred)
        res = {
            'auc': auc_score,
            'ap': ap_score
        }
    elif args.task == "gc":
        y_pred = clf.predict(test_emb)        
        input_dict = {'y_pred': y_pred, 'y_true': labels[test_ids]}        
        labels = labels.reshape(-1, 1)
        onehot_encoder = OneHotEncoder(categories='auto').fit(labels)
        labels = onehot_encoder.transform(labels).toarray().astype(np.bool_)        
        test_labels = np.nonzero(labels[test_ids])[1]
        micro = f1_score(test_labels, y_pred, average="micro")
        macro = f1_score(test_labels, y_pred, average="macro")
        acc = accuracy_score(test_labels, y_pred)
        res = {
            'F1Mi': micro,
            'F1Ma': macro,
            'acc': acc}
    return res


def train(args, log):    
    tmp_save_dir = os.path.join(sys.path[0], 'tmp', f'{args.data}')
    if not os.path.exists(tmp_save_dir):
        os.makedirs(tmp_save_dir)             
    graphs, (num_features, num_classes) = load_graph_classification_dataset(args.data)                                    
    train_idx = torch.arange(len(graphs))
    train_sampler = SubsetRandomSampler(train_idx)     
    dataloader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_gc, batch_size=args.batch, pin_memory=True)
    test_dataloader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=args.batch, shuffle=False)
 

    adv_loss_fun = nn.BCELoss(reduction='mean')
   
    gen_model = build_gen_model(args, num_features).to(args.device)
    gen_opt = create_optimizer(args.gen_optimizer, gen_model, args.lr, args.weight_decay)

    # discriminator
    dis_model = build_dis_model(args, num_features).to(args.device)
    dis_opt = create_optimizer(args.dis_optimizer, dis_model, args.lr, args.weight_decay)

    for epoch in tqdm(range(args.epochs)):
        gen_model.train()
        dis_model.train()
                
        ###################### discriminator########################
        for idx, batched_graph in enumerate(dataloader):            

            gen_batch_graph, masked_nodes, _, _ = gen_model(batched_graph, args.device)            
            # real dis loss
            dis_real_logits = dis_model(batched_graph, args.device).squeeze(-1)
            dis_real_size = dis_real_logits.shape[0]
            dis_labels_real = torch.tensor([1] * dis_real_size).float()
            dis_labels_real = dis_labels_real.to(args.device)
            loss_dis_real = adv_loss_fun(dis_real_logits, dis_labels_real)            
            
            # fake dis loss                         
            dis_fake_logits = dis_model(gen_batch_graph, args.device).squeeze(-1)
            dis_fake_size = dis_fake_logits.shape[0]                         
            dis_labels_fake = torch.tensor([0] * dis_fake_size).float()
            dis_labels_fake = dis_labels_fake.to(args.device)
            loss_dis_fake = adv_loss_fun(dis_fake_logits, dis_labels_fake)

            loss_dis = (loss_dis_real + loss_dis_fake) / 2                        
            
            dis_opt.zero_grad()                                         
            loss_dis.backward(retain_graph=True)            
            dis_opt.step()
 
            ###################### generator########################       
            rec_loss = rec_loss_fun(gen_batch_graph, batched_graph, masked_nodes, args.device)

            dis_logits = dis_model(gen_batch_graph, args.device).squeeze(-1)
            gen_labels_real = torch.tensor([1] * dis_logits.shape[0]).float().to(args.device)
            dis_loss = adv_loss_fun(dis_logits, gen_labels_real)   
             
            loss_gen = rec_loss + args.alpha * dis_loss 
            gen_opt.zero_grad()
            loss_gen.backward()
            gen_opt.step()
                        
        
        if epoch % 20 == 0:                      
            test_f1 = graph_classification_evaluation(gen_model, test_dataloader,  args.device, mute=False) 
            logging.info("Epoch {:05d} |  D:loss_dis {:.4f}| G: loss_gen {:.4f}|  G:rec_loss {:.4f}|  G:dis {:.4f} | test_Acc {:.4f}"
                        .format(epoch, loss_dis, loss_gen, rec_loss, dis_loss, test_f1))            
    test_f1 = graph_classification_evaluation(gen_model, test_dataloader,  args.device, mute=False)    
    return test_f1


def get_result(res_list):
    all_keys = set().union(*res_list) 
    mean_std = {}

    for key in all_keys:        
        values = [d[key] for d in res_list if key in d]        
        mean = np.mean(values)
        std = np.std(values)
        
        mean_std[f"{key}_mean"] = mean
        mean_std[f"{key}_std"] = std

    return mean_std


def main():    
    now = datetime.datetime.now()
    nowtime = now.strftime("%Y_%m_%d_%H_%M_%S")
    args = read_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")             

    args.device = f'cuda:{args.device}'
    filename = "debug"+str(args.debug) + '_' + args.data + \
        '_' + args.task+'_' + nowtime + '.log'
    log = Logger(filename, args.data, level='info')
    log = log.logger
    log.info(args)
    for k, v in vars(args).items():
        log.info(k + ':' + str(v))
    accs = []
    for i in range(args.repeats):
        res = train(args, log)
        accs.append(res)
        
    message = {}    
    acc_mean = np.round(np.mean(accs), 5)
    acc_std = np.round(np.std(accs), 5)
    message['acc_mean'] = acc_mean
    message['acc_std'] = acc_std
    log.info(message)
    
if __name__ == "__main__":
    args = read_args()    
    main()
    