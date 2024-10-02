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
Task3: Load 'PubMed' dataset with 'public' split.
       Build a 2-layer GraphSAGE model with 'max' aggregator.
       Sample 25 and 10 nodes for the 1st and 2nd layer, respectively. Use 'batch_size=128'.
       Train and test the model on 'PubMed' datasets.
       Take a screenshot of the output and include the screenshot in your report.

       Note: You only need to complete the missing code (indicated by ##TODO). DO NOT change other parts of the code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric


## Load PubMed dataset
## https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid
## PubMed dataset: a citation network with 19717 scientific publications classified into one of 3 classes.
## Nodes (19717) mean scientific publications and edges (88648) mean citation relationships. Each node has a predefined feature with 500 dimensions.
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

## Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g = g.to(device)

## Define sampler and dataloader
## https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborSampler
##TODO
##===
train_loader =

##===
test_loader = torch_geometric.loader.NeighborSampler(g.edge_index, node_idx=None, sizes=[-1],
                                batch_size=128, shuffle=False, num_workers=0)

from torch_geometric.nn import SAGEConv

class Minibatch_GraphSAGE(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, num_classes):
        super(Minibatch_GraphSAGE, self).__init__()
        ## 2-layer GraphSAGE model
        self.num_layers = 2
        self.layers = torch.nn.ModuleList()
        ##TODO
        ##=== Append two SAGEConv layers with 'max' aggregator.


        ##===

    def forward(self, h, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            h_target = h[:size[1]]
            h = self.layers[i]((h, h_target), edge_index)
            if i != self.num_layers - 1:
                h = F.relu(h)
        return h.log_softmax(dim=-1)

    def inference(self, h_all):
        for i in range(self.num_layers):
            hs = []
            for _, n_id, adj in test_loader:
                edge_index, _, size = adj.to(device)
                h = h_all[n_id].to(device)
                h_target = h[:size[1]]
                h = self.layers[i]((h, h_target), edge_index)
                if i != self.num_layers - 1:
                    h = F.relu(h)
                hs.append(h.log_softmax(dim=-1))

            h_all = torch.cat(hs, dim=0)

        return h_all

def Minibatch_train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    model.train()
    for epoch in range(1, 101):

        for batch_size, n_id, adjs in train_loader:
            adjs = [adj.to(device) for adj in adjs]
            ## Forward
            out = model(g.x[n_id], adjs)
            ## Compute loss
            loss = F.cross_entropy(out, g.y[n_id[:batch_size]])
            ## Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ## Compute prediction
        model.eval()
        preds = model.inference(g.x).argmax(dim=1)

        ## Compute accuracy on training/validation/test
        train_acc = (preds[g.train_mask] == g.y[g.train_mask]).float().mean()
        val_acc = (preds[g.val_mask] == g.y[g.val_mask]).float().mean()
        test_acc = (preds[g.test_mask] == g.y[g.test_mask]).float().mean()

        ## Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 10 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                epoch, loss, val_acc, best_val_acc, test_acc, best_test_acc))

## Create the model with given dimensions
model = Minibatch_GraphSAGE(dataset.num_node_features, 16, dataset.num_classes).to(device)

## Train the model
Minibatch_train(g, model)
