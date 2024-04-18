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
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.svm import SVC
from sklearn.metrics import f1_score,accuracy_score



def graph_classification_evaluation(model, dataloader, device, mute=False):
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, (batch_g, labels) in enumerate(dataloader):
            batch_g = batch_g.to(device)
            feat = batch_g.ndata["attr"]
            out = model.embed(batch_g, feat)

            batch_g.ndata['h'] = out
            out = dgl.mean_nodes(batch_g, 'h')
            # out = pooler(batch_g, out)

            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
    return test_f1


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    accs = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        # f1 = f1_score(y_test, preds, average="micro")
        acc = accuracy_score(y_test, preds)
        # result.append(f1)
        accs.append(acc)
    # test_f1 = np.mean(result)
    # test_std = np.std(result)

    acc_mean = np.mean(accs)
    acc_std = np.std(accs)
        
    return acc_mean, acc_std