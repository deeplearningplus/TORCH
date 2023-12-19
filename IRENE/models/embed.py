# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

import torch.nn as nn
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs
from models.attention import Attention
import pdb

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, input_size):
        super(Embeddings, self).__init__()

        self.max_patches =  1024 ##

        self.patch_embeddings = nn.Linear(input_size, config.hidden_size, bias=False)
        self.sex_embeddings = nn.Embedding(2, config.hidden_size)   # Male, Female
        self.age_embeddings = nn.Embedding(120, config.hidden_size) # Age, from 0 to 120 years 
        self.origin_embeddings = nn.Embedding(2, config.hidden_size)   # Male, Female
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1+self.max_patches, config.hidden_size))
        self.pe_sex = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_age = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_origin = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = nn.Dropout(config.transformer["dropout_rate"])
        self.dropout_sex = nn.Dropout(config.transformer["dropout_rate"])
        self.dropout_age = nn.Dropout(config.transformer["dropout_rate"])
        self.dropout_origin = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x, sex, age, origin):
        # x is the embedded image features, batch_size x n_patches x n_features
        B, N, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x) 
        sex = self.sex_embeddings(sex)
        age = self.age_embeddings(age)
        origin = self.origin_embeddings(origin)

        #x = x.flatten(2)
        #x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings[:,:(N + 1)] # https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
        sex_embeddings = sex + self.pe_sex
        age_embeddings = age + self.pe_age
        origin_embeddings = origin + self.pe_origin

        embeddings = self.dropout(embeddings)
        sex_embeddings = self.dropout_sex(sex_embeddings)
        age_embeddings = self.dropout_age(age_embeddings)
        origin_embeddings = self.dropout_origin(origin_embeddings)
        return embeddings, sex_embeddings, age_embeddings, origin_embeddings



