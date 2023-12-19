import argparse
parser = argparse.ArgumentParser(description='Training script for IRENE model')
parser.add_argument('--train-file', help='training set file formatted as CSV that have these columns: cyto_pt_file, histo_pt_file, age, sex, origin, label')
parser.add_argument('--val-file', help='validation set file formatted as CSV (the same as above)')
parser.add_argument('--input-dim', help='the dim of extracted image-patch features (default: 2048)', default=2048, type=int)
parser.add_argument('--outdir', help='output directory')
parser.add_argument('--num-classes', help='number of classes (default: 5)', default=5, type=int)
parser.add_argument('--cyto-pt-file', help='column name refers to the extracted cytological features')
parser.add_argument('--histo-pt-file', help='column name refers to the extracted histological features')
parser.add_argument('--device', help='device to train the model (default: cuda:0)', default='cuda:0', type=str)
parser.add_argument('--disable-tqdm', help='disable tqdm progress bar', default=False, choices=[False, True], type=bool)
parser.add_argument('--mode', help='use either cytological, histological or both features', default="both", choices=["cyto", "histo", "both"], type=str)
args = parser.parse_args()

import os
import sys
import torch
from torch.optim import Adam, AdamW, lr_scheduler
from models.modeling_irene import IRENE, CONFIGS
from tqdm import tqdm
import pandas as pd
import random
import cv2

torch.set_num_threads(1)
cv2.setNumThreads(1)


config = CONFIGS["IRENE"]
config.hidden_size = 384
config.transformer.mlp_dim = config.hidden_size * 4
config.transformer.num_heads = 6
config.transformer.num_layers = 3
config.transformer.attention_dropout_rate = 0.1
config.transformer.dropout_rate = 0.1
print(config)
net = IRENE(config, input_size=args.input_size, num_classes=args.num_classes, vis=False)
print(net)

best_acc = 0.0
epochs = 100
device = args.device


net = net.to(device)
opt = Adam(net.parameters(), lr=2e-4, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(opt, step_size=100, gamma=0.1)

trn_df = pd.read_csv(args.train_file)
val_df = pd.read_csv(args.val_file)

fout = open(f'{args.outdir}/log.txt', 'w')
print('epoch\ttrain_loss\ttrain_acc\teval_loss\teval_acc', file=fout)

def parse_data_v2(cyto_pt_file, histo_pt_file, age, sex, origin, label, device):
    x1 = torch.load(cyto_pt_file)
    x2 = torch.load(histo_pt_file)
    x = torch.cat([x1, x2], dim=0)
    x = x.unsqueeze(0).to(device)
    age = torch.tensor([age]).to(device)
    sex = torch.tensor([sex]).to(device)
    origin = torch.tensor([origin]).to(device)
    y = torch.tensor([label]).to(device)
    return x, age, sex, origin, y

def parse_data_v1(pt_file, age, sex, origin, label, device):
    x = torch.load(pt_file)
    x = x.unsqueeze(0).to(device)
    age = torch.tensor([age]).to(device)
    sex = torch.tensor([sex]).to(device)
    origin = torch.tensor([origin]).to(device)
    y = torch.tensor([label]).to(device)
    return x, age, sex, origin, y

for epoch in range(epochs):
    net.train()
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

        out = net(x=x, sex=sex, age=age, origin=origin, labels=y)
        loss = out["loss"]
        logits = out["logits"]

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        correct_k += sum(y.eq(logits.argmax(1)))

    train_loss = total_loss / len(trn)
    train_acc = correct_k / len(trn)

    scheduler.step()

    net.eval()
    total_loss = 0.0
    correct_k = 0.0
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

        out = net(x=x, sex=sex, age=age, origin=origin, labels=y)
        loss = out["loss"]
        logits = out["logits"]

        total_loss += loss.item()
        correct_k += sum(y.eq(logits.argmax(1)))


    val_loss = total_loss / len(val)
    val_acc = correct_k / len(val)

    print(f"{epoch + 1}\t{train_loss}\t{train_acc}\t{val_loss}\t{val_acc}", file=fout)
    sys.stdout.flush()
    fout.flush()

    if val_acc >= 0.65 or epoch + 1 >= 30:
        torch.save(net.state_dict(), f'{args.outdir}/model_{epoch+1}.pt')
        best_acc = val_acc

fout.close()


