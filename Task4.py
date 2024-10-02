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
Task4: Load 'CiteSeer' dataset with 'full' split.
       Tune the model hyperparameters to obtain a possibly best val_acc on 'CiteSeer' dataset.
       Take a screenshot of the best results and include the screenshot in your report. Also, include the choices of hyperparameters in your report as well.
       You should achieve val_acc > 0.77.
       The student who obtains the highest val_acc will get extra credit.

       Note: You only need to complete the missing code (indicated by ##TODO). DO NOT change other parts of the code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

torch.manual_seed(1)


## Load CiteSeer dataset
## https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid
## CiteSeer dataset: a citation network with 3327 scientific publications classified into one of 6 classes.
## Nodes (3327) mean scientific publications and edges (9104) mean citation relationships. Each node has a predefined feature with 3703 dimensions.
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


from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, num_classes, num_layers, aggregator):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(in_dimension, hidden_dimension, aggr=aggregator))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dimension, hidden_dimension, aggr=aggregator))
        self.layers.append(SAGEConv(hidden_dimension, num_classes, aggr=aggregator))

    def forward(self, g):
        h, edge_index = g.x, g.edge_index
        for _, layer in enumerate(self.layers[:-1]):
            h = layer(h, edge_index)
            h = F.relu(h)
        h = self.layers[-1](h, edge_index)
        return F.log_softmax(h, dim=1)


def train(g, model, lr, n_epoch):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_test_acc = 0

    model.train()
    for epoch in range(1, n_epoch+1):
        out = model(g)
        loss = F.cross_entropy(out[g.train_mask], g.y[g.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        pred = model(g).argmax(dim=1)

        train_acc = (pred[g.train_mask] == g.y[g.train_mask]).float().mean()
        val_acc = (pred[g.val_mask] == g.y[g.val_mask]).float().mean()
        test_acc = (pred[g.test_mask] == g.y[g.test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 1 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                epoch, loss, val_acc, best_val_acc, test_acc, best_test_acc))


device = torch.device('cpu')
g = g.to(device)

##TODO (Tune the hyperparameters below)
##===
# Set number of SAGEConv layers. You can choose any number >= 2.
num_layers = 2
# Set hidden dimension. You can choose any positive number.
hidden_dimension = 8
# Set aggregator. You can choose 'mean' or 'max'.
aggregator = 'mean'
# Set learning rate. You can choose any positive number.
learning_rate = 0.1
# Set training epoch. You can choose any positive number.
number_epoch = 20
##===

model = GraphSAGE(dataset.num_node_features, hidden_dimension, dataset.num_classes, num_layers, aggregator).to(device)

train(g, model, learning_rate, number_epoch)
