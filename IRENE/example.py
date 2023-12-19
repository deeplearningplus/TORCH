#!/opt/software/install/miniconda37/bin/python

import torch
from models.modeling_irene import IRENE, CONFIGS
import sys
from prettytable import PrettyTable

# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/22?page=2
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters (Mb)"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel() / 1e6
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


config = CONFIGS["IRENE"]
config.hidden_size = 512
config.transformer.mlp_dim = config.hidden_size * 4
config.transformer.num_heads = 8
config.transformer.num_layers = 6

input_size = 2048 # the extracted image feature dim

model = IRENE(config, input_size=input_size, zero_head=True, num_classes=5, vis=False)
count_parameters(model)

x = torch.randn(1, 80, 2048) # An image of 80 patches with each patch projected into a feature of 768 dims.
sex = torch.tensor([1])     # sex info, 0: male, 1: female
age = torch.tensor([65])    # age info, an integer, range from 0 to 120
labels = torch.tensor([3])  # class label, an integer

out = model(x, sex, age, labels=labels)
print(out);sys.exit(1)



# attn_weights is empty is vis=False when initialized IRENE model.
logits, attn_weights, x_hat = out 

print(logits)
print(attn_weights)
print(x_hat.shape)


