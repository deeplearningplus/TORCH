import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, input_dim = 1024, gate = True, size_arg = "big", dropout = True, n_classes = 2):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [input_dim, 512, 256], "big": [input_dim, 512, 384]}
        size = self.size_dict[size_arg]

        self.sex_embedding = nn.Embedding(2, size[0])    # Male: 0, Female: 1
        self.age_embedding = nn.Embedding(100, size[0])  # Age: integer value
        self.origin_embedding = nn.Embedding(2, size[0]) # 腹水(Ascites): 0, 胸水(pleural effusion): 1

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes

        initialize_weights(self)

    def forward(self, h, sex=None, age=None, origin=None, return_features=False, attention_only=False):
        sex_emb = self.sex_embedding(sex)
        age_emb = self.age_embedding(age)
        origin_emb = self.origin_embedding(origin)
        h = torch.cat((h, sex_emb, age_emb, origin_emb), dim=0)

        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        #A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h) 
        if return_features:
            return M

        logits = self.classifiers(M)
        #Y_hat = torch.topk(logits, 1, dim = 1)[1]
        #Y_prob = F.softmax(logits, dim = 1)

        return logits

class CLAM_MB(nn.Module):
    def __init__(self, input_dim = 1024, gate = True, size_arg = "big", dropout = True, n_classes = 2):
        super(CLAM_MB, self).__init__()
        self.size_dict = {"small": [input_dim, 512, 256], "big": [input_dim, 512, 384]}
        size = self.size_dict[size_arg]

        self.sex_embedding = nn.Embedding(2, size[0])    # Male: 0, Female: 1
        self.age_embedding = nn.Embedding(100, size[0])  # Age: integer value
        self.origin_embedding = nn.Embedding(2, size[0]) # 腹水(Ascites): 0, 胸水(pleural effusion): 1

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        self.n_classes = n_classes
        initialize_weights(self)

    def forward(self, h, sex=None, age=None, origin=None, return_features=False, attention_only=False):
        sex_emb = self.sex_embedding(sex)
        age_emb = self.age_embedding(age)
        origin_emb = self.origin_embedding(origin)
        h = torch.cat((h, sex_emb, age_emb, origin_emb), dim=0)

        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        #A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h) 
        if return_features:
            return M

        logits = torch.empty(1, self.n_classes).float()
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        #Y_hat = torch.topk(logits, 1, dim = 1)[1]
        #Y_prob = F.softmax(logits, dim = 1)

        return logits
