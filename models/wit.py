import torch
import torch.nn as nn
import transformers

def exists(val):
    return val is not None

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

class WIT(nn.Module):

    def __init__(self, input_dim=1024, n_classes=2):
        super(WIT, self).__init__()

        self.proj_in = nn.Linear(input_dim, 384, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 384))
        self.sex_embedding = nn.Embedding(2, 384)    # Male: 0, Female: 1
        self.age_embedding = nn.Embedding(100, 384)  # Age: integer value
        self.origin_embedding = nn.Embedding(2, 384) # 腹水(Ascites): 0, 胸水(pleural effusion): 1
        
        self.wit = transformers.BertForSequenceClassification(
                transformers.BertConfig(hidden_size=384, intermediate_size=1536,
                    num_attention_heads=6, num_hidden_layers=1, vocab_size=2, num_labels=n_classes))

        initialize_weights(self)
        
    def forward(self, h, sex=None, age=None, origin=None, return_features=False, attention_only=False):
        h = self.proj_in(h)

        if exists(sex) and exists(age) and exists(origin):
            sex_emb = self.sex_embedding(sex) 
            age_emb = self.age_embedding(age)
            origin_emb = self.origin_embedding(origin)
            h = torch.cat((self.cls_token, h, sex_emb, age_emb, origin_emb), dim=0)
        else:
            h = torch.cat((self.cls_token, h), dim=0)
        h = h.unsqueeze(0)
        b, n, _ = h.shape

        logits = self.wit(inputs_embeds=h).logits

        return logits

