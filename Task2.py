"""
Node Classification with GNN for CS 7315

Copyright (c) 2022 Texas State University. All rights reserved.

Redistribution in source and binary forms, with or without modification, is not
permitted. Use in source or binary form, with or without modification, is only
permitted for academic use in CS 7315 at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Xin Huang

"""


"""
Task2: Load 'Cora' dataset with 'public' split.
       Build a 3-layer GCN model. Train and test the model on 'Cora' dataset.
       Take a screenshot of the output and include the screenshot in your report.

       Note: You only need to complete the missing code (indicated by ##TODO). DO NOT change other parts of the code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric


## Load Cora dataset
## https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid
## Cora dataset: a citation network with 2708 scientific publications classified into one of 7 classes.
## Nodes (2708) mean scientific publications and edges (10556) mean citation relationships. Each node has a predefined feature with 1433 dimensions.
##TODO
##===

##===
g = dataset[0]

print(f'Number of nodes: {g.num_nodes}')
print(f'Number of edges: {g.num_edges}')
print(f'Number of features: {dataset.num_node_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of training nodes: {g.train_mask.sum()}')
print(f'Number of validation nodes: {g.val_mask.sum()}')
print(f'Number of test nodes: {g.test_mask.sum()}')
print(f'Has self loops?: {g.has_self_loops()}')
print(f'Is directed?: {g.is_directed()}')


## Use build-in GCNConv layer
## https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, num_classes):
        super(GCN, self).__init__()
        ##TODO
        ##=== 3-layer GCN model



        ##===

    def forward(self, g):
        h, edge_index = g.x, g.edge_index
        ##TODO
        ##=== Apply F.relu to 1st and 2nd layer.






        ##===
        return F.log_softmax(h, dim=1)


## Define training process
def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ## Record best accuracy
    best_val_acc = 0
    best_test_acc = 0

    model.train()
    for epoch in range(1, 11):
        ## Forward
        out = model(g)
        ## Compute loss
        loss = F.cross_entropy(out[g.train_mask], g.y[g.train_mask])
        ## Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ## Compute prediction
        model.eval()
        pred = model(g).argmax(dim=1)
        ## Compute accuracy on training/validation/test
        train_acc = (pred[g.train_mask] == g.y[g.train_mask]).float().mean()
        val_acc = (pred[g.val_mask] == g.y[g.val_mask]).float().mean()
        test_acc = (pred[g.test_mask] == g.y[g.test_mask]).float().mean()

        ## Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 1 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                epoch, loss, val_acc, best_val_acc, test_acc, best_test_acc))


## Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g = g.to(device)

## Create the model with given dimensions
model = GCN(dataset.num_node_features, 64, dataset.num_classes).to(device)

## Train the model
train(g, model)
