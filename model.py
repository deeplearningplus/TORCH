import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(val):
    return val is not None

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, n_tasks = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Dropout(0.25))
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(0.25))
        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
TORCH - Tumor Origin diffeRentiation for Cytologic Histology
Code borrowed from: Lu et al. AI-based pathology predicts origins for cancers of unknown primary. Nature 2020.
args:
    input_dim: the input dimension
    size_args: size config of attention network
    n_classes: number of classes
"""

class TORCH(nn.Module):

    def __init__(self, input_dim = 1024, size_arg = "big", n_classes = 2):
        super(TORCH, self).__init__()
        self.size_dict = {"small": [input_dim, 512, 256], "big": [input_dim, 512, 384]}
        size = self.size_dict[size_arg]

        self.sex_embedding = nn.Embedding(2, size[1])    # Male: 0, Female: 1
        self.age_embedding = nn.Embedding(120, size[1])  # Age: integer value
        self.origin_embedding = nn.Embedding(2, size[1]) # 腹水(Ascites): 0, 胸水(pleural effusion): 1

        self.attention_net = nn.Sequential(
            nn.Linear(size[0], size[1]), 
            nn.ReLU(), 
            nn.Dropout(0.25),
            Attn_Net_Gated(L = size[1], D = size[2], n_tasks = 1))
        
        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)
        
    def forward(self, h, sex=None, age=None, origin=None, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A[0]
        
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)

        if exists(sex):
            M = M + self.sex_embedding(sex) 
        if exists(age):
            M = M + self.age_embedding(age)
        if exists(origin):
            M = M + self.origin_embedding(origin)

        if return_features:
            return M

        logits  = self.classifier(M)

        return logits


