import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from layers import GraphAttentionLayer


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = f.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = f.dropout(x, self.dropout, training=self.training)
        x = f.elu(self.out_att(x, adj))
        return x
        # return F.log_softmax(x, dim=1)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_gat_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(CriticNetwork, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""
        self.gat = GAT(nfeat=input_dim,
                    nhid=hidden_gat_dim,
                    nclass=input_dim,
                    dropout=0.0,
                    nheads=4,
                    alpha=0.1)

        # self.gat2 = GAT(nfeat=input_dim,
        #             nhid=hidden_gat_dim,
        #             nclass=input_dim,
        #             dropout=0.0,
        #             nheads=4,
        #             alpha=0.1)

        dense_input_dim = input_dim * 5 * 2 # num_agents * res connections
        self.fc1 = nn.Linear(dense_input_dim, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.nonlin = f.relu  # leaky_relu
        self.actor = actor
        # self.reset_parameters()

    def reset_parameters(self):
        self.gat.weight.data.uniform_(*hidden_init(self.gat))
        # self.gat2.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x, adj):
        gat = self.gat(x, adj)
        resgat = torch.cat((x, gat), dim=-1)  # review
        flatten = torch.flatten(resgat, start_dim=1)
        h1 = self.nonlin(self.fc1(flatten))
        h2 = self.nonlin(self.fc2(h1))
        h3 = (self.fc3(h2))
        return h3


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(ActorNetwork, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.nonlin = f.relu  # leaky_relu
        self.actor = actor
        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        # return a vector of the force
        h1 = self.nonlin(self.fc1(x))
        h2 = self.nonlin(self.fc2(h1))
        h3 = (self.fc3(h2))
        norm = torch.norm(h3)

        # h3 is a 2D vector (a force that is applied to the agent)
        # we bound the norm of the vector to be between 0 and 10
        # return 10.0*(torch.tanh(norm))*h3/norm if norm > 0 else 10*h3
        return 1.0 * (torch.tanh(norm)) * h3 / norm if norm > 0 else 1 * h3
