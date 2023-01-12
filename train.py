#!/usr/bin/env python
import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from model import TORCH
import pandas as pd

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="TORCH training", add_help=add_help)
    parser.add_argument("--outdir", type=str, help="output directory")
    parser.add_argument("--device", default='cuda:0', type=str, help="CUDA device (default: cuda)")
    parser.add_argument("--epochs", default=100, type=str, help="training epochs (default: 100)")
    parser.add_argument("--num_classes", type=int, help="number of classes")
    parser.add_argument("--train_file", type=str, help="training file")
    parser.add_argument("--val_file", type=str, help="validation file")
    return parser

args = get_args_parser().parse_args()

outdir = args.outdir
device = args.device
epochs = args.epochs
num_classes = args.num_classes
train_file = args.train_file
val_file = args.val_file


trn = pd.read_csv(train_file)
val = pd.read_csv(val_file)

fout = open(f'{outdir}/loss.txt', 'w')
print('epoch\ttrain_loss\ttrain_acc\teval_loss\teval_acc', file=fout)

model = TORCH(input_dim=2048, n_classes=num_classes)
model = model.to(device)
print(model)


criterion = nn.CrossEntropyLoss()
opt = Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

for epoch in range(epochs):
    model.train()
    total_loss, correct_k = 0, 0
    trn = trn.sample(len(trn))

    ## Training
    for pt, label, age, sex, origin in zip(trn.pt, trn.label, trn.age, trn.sex, trn.origin):
        age, sex, origin = torch.tensor([age]), torch.tensor([sex]), torch.tensor([origin])
        y = torch.as_tensor([label])
        x = torch.load(pt)
        y, x = y.to(device), x.to(device)
        age, sex, origin = age.to(device), sex.to(device), origin.to(device)

        logit = model(x, sex, age, origin)     
        loss = criterion(logit, y) 

        loss.backward()
        opt.step()
        opt.zero_grad()

        total_loss += loss.item()
        correct_k += logit.argmax(1).eq(y).sum()

    train_acc = correct_k / len(trn)
    train_loss = total_loss / len(trn)


    ## Evaluating
    model.eval()
    total_loss, correct_k = 0, 0
    for pt, label, age, sex, origin in zip(val.pt, val.label, val.age, val.sex, val.origin):
        age, sex, origin = torch.tensor([age]), torch.tensor([sex]), torch.tensor([origin])
        y = torch.as_tensor([label])
        x = torch.load(pt)
        y, x = y.to(device), x.to(device)
        age, sex, origin = age.to(device), sex.to(device), origin.to(device)

        with torch.no_grad():
            logit = model(x, sex, age, origin)     
    
        loss = criterion(logit, y) 

        total_loss += loss.item()
        correct_k += logit.argmax(1).eq(y).sum()

    eval_acc = correct_k / len(val)
    eval_loss = total_loss / len(val)

    print(f"Epoch: {epoch + 1}; train loss: {train_loss:.5f}, acc: {train_acc:.5f}; eval loss: {eval_loss:.5f}, acc: {eval_acc:.5f}; ")
    sys.stdout.flush()
    print(f"{epoch + 1}\t{train_loss}\t{train_acc}\t{eval_loss}\t{eval_acc}", file=fout)
    fout.flush()

    torch.save(model.state_dict(), f'{outdir}/model_{epoch+1}.pt')

fout.close()


