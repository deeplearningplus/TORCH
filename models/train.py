#!/opt/software/install/miniconda37/bin/python
import argparse
parser = argparse.ArgumentParser(description='Training for Attention-based MIL')
parser.add_argument('--train-file', help='training set file formatted as CSV that have these columns: cyto_pt_file, histo_pt_file, age, sex, origin, label')
parser.add_argument('--val-file', help='validation set file formatted as CSV (the same as above)')
parser.add_argument('--input-dim', help='the dim of extracted image-patch features (default: 2048)', default=2048, type=int)
parser.add_argument('--outdir', help='output directory')
parser.add_argument('--model-name', help='model name (default: AttnMIL)', default='AttnMIL', choices=['AttnMIL', 'CLAM_MB', 'WIT'], type=str)
parser.add_argument('--num-classes', help='number of classes (default: 5)', default=5, type=int)
parser.add_argument('--cyto-pt-file', help='column name refers to the extracted cytological features')
parser.add_argument('--histo-pt-file', help='column name refers to the extracted histological features')
parser.add_argument('--device', help='device to train the model (default: cuda:0)', default='cuda:0', type=str)
parser.add_argument('--disable-tqdm', help='disable tqdm progress bar', default=False, choices=[False, True], type=bool)
parser.add_argument('--mode', help='use either cytological, histological or both features', default="both", choices=["cyto", "histo", "both"], type=str)
args = parser.parse_args()

import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW, Adam, SGD
import numpy as np
import pandas as pd
import cv2

from wit import WIT
from model import TORCH
from model_clam import CLAM_MB
from model_dual_attentions import TORCH_dual_attentions

torch.set_num_threads(1)
cv2.setNumThreads(1)

device = args.device #"cuda:0"
epochs = 100
img_size = 1024


if args.model_name == 'AttnMIL':
    model = TORCH(input_dim=args.input_dim, n_classes=args.num_classes, size_arg="big")
elif args.model_name == 'WIT':
    model = WIT(input_dim=args.input_dim, n_classes=args.num_classes)
#elif args.model_name == 'TORCH_dual_attentions':
#    model = TORCH_dual_attentions(input_dim=args.input_dim, n_classes=args.num_classes, size_arg="big")
elif args.model_name == 'CLAM_MB':
    model = CLAM_MB(input_dim=args.input_dim, n_classes=args.num_classes)

model = model.to(device)
print(model)


def parse_data_v2(cyto_pt_file, histo_pt_file, age, sex, origin, label, device):
    x1 = torch.load(cyto_pt_file)
    x2 = torch.load(histo_pt_file)
    x = torch.cat([x1, x2], dim=0)
    #x = x.unsqueeze(0).to(device)
    x = x.to(device)
    age = torch.tensor([age]).to(device)
    sex = torch.tensor([sex]).to(device)
    origin = torch.tensor([origin]).to(device)
    y = torch.tensor([label]).to(device)
    return x, age, sex, origin, y

def parse_data_v1(pt_file, age, sex, origin, label, device):
    x = torch.load(pt_file)
    #x = x.unsqueeze(0).to(device)
    x = x.to(device)
    age = torch.tensor([age]).to(device)
    sex = torch.tensor([sex]).to(device)
    origin = torch.tensor([origin]).to(device)
    y = torch.tensor([label]).to(device)
    return x, age, sex, origin, y

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

best_loss = 1000.0
trn_df = pd.read_csv(args.train_file)
val_df = pd.read_csv(args.val_file)

fout = open(f'{args.outdir}/log.txt', 'w')
print('epoch\ttrain_loss\ttrain_acc\teval_loss\teval_acc', file=fout)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct_k = 0

    trn = trn_df.sample(len(trn_df))
    trn_iters = tqdm(
            zip(range(len(trn)), trn[args.cyto_pt_file], trn[args.histo_pt_file], trn.age, trn.sex, trn.origin, trn.label), 
            disable=args.disable_tqdm, 
            total=len(trn)
        )

    for i, cyto_pt_file, histo_pt_file, age, sex, origin, label in trn_iters:
        if args.mode == "both":
            x, age, sex, origin, y = parse_data_v2(cyto_pt_file, histo_pt_file, age, sex, origin, label, device)
        elif args.mode == "cyto":
            x, age, sex, origin, y = parse_data_v1(cyto_pt_file, age, sex, origin, label, device)
        elif args.mode == "histo":
            x, age, sex, origin, y = parse_data_v1(histo_pt_file, age, sex, origin, label, device)

        logit = model(h=x, sex=sex, age=age, origin=origin)

        if logit.device != label.device:##CLAM_MB
            logit = logit.to(label.device)

        loss = criterion(logit, label) 

        optimizer.step()
        loss.backward()
        optimizer.zero_grad()

        total_loss += loss.item()
        correct_k += logit.argmax(1).eq(label).sum()

    train_acc = correct_k / total_num
    train_loss = total_loss / total_batch


    model.eval()
    total_loss = 0.0
    correct_k = 0

    val = val_df
    val_iters = tqdm(
            zip(range(len(val)), val[args.cyto_pt_file], val[args.histo_pt_file], val.age, val.sex, val.origin, val.label), 
            disable=args.disable_tqdm, 
            total=len(val)
        )

    for i, cyto_pt_file, histo_pt_file, age, sex, origin, label in val_iters:
        if args.mode == "both":
            x, age, sex, origin, y = parse_data_v2(cyto_pt_file, histo_pt_file, age, sex, origin, label, device)
        elif args.mode == "cyto":
            x, age, sex, origin, y = parse_data_v1(cyto_pt_file, age, sex, origin, label, device)
        elif args.mode == "histo":
            x, age, sex, origin, y = parse_data_v1(histo_pt_file, age, sex, origin, label, device)

        with torch.no_grad():
            logit = model(h=x, sex=sex, age=age, origin=origin)
    
        if logit.device != label.device:
            logit = logit.to(label.device)

        loss = criterion(logit, label)
        total_loss += loss.item()
        correct_k += logit.argmax(1).eq(label).sum()

    val_loss = total_loss / len(val)
    val_acc = correct_k / len(val)

    print(f"{epoch + 1}\t{train_loss}\t{train_acc}\t{val_loss}\t{val_acc}", file=fout)
    fout.flush()

    if val_acc >= 0.65 or epoch + 1 >= 30:
        torch.save(model.state_dict(), f'{args.outdir}/model_{epoch+1}.pt')

fout.close()


