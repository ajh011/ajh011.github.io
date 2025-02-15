import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import aggr

class CGConv(MessagePassing):
    def __init__(self, node_fea_dim=92, edge_fea_dim=80, batch_norm = True):
        super().__init__()
        self.batch_norm = batch_norm
        self.node_fea_dim = node_fea_dim
        self.edge_fea_dim = edge_fea_dim

        self.lin_full = Linear(2*node_fea_dim+edge_fea_dim, 2*node_fea_dim)

        self.sigmoid_filter = torch.nn.Sigmoid()
        self.softplus_core = torch.nn.Softplus()
        self.softplus_out = torch.nn.Softplus()

        self.node_aggr = aggr.SumAggregation()
       
        if batch_norm == True:
            self.bn_i = BatchNorm1d(2*node_fea_dim)
            self.bn_o = BatchNorm1d(node_fea_dim)
    
    def forward(self, x, edge_index, edge_attr):
        origin_x = x[edge_index[0]]
        node_nbr_x = x[edge_index[1]]
        total_message = torch.cat([origin_x, edge_attr, node_nbr_x], dim=1)
        total_message = self.lin_full(total_message)
        if self.batch_norm == True:
            total_message = self.bn_i(total_message)
        filter, core = total_message.chunk(2, dim=1)
        gated_message = self.sigmoid_filter(filter)*self.softplus_core(core)
        out_message = self.node_aggr(gated_message, edge_index[0])
        if self.batch_norm == True:
            out_message = self.bn_o(out_message)
        out_x = self.softplus_out(x + out_message)

        return out_x
    
class CGCNN(nn.Module):
    def __init__(self, x_dim, edge_dim,
                 h_dim=64, n_conv=3, hout_dim=128, n_h=1,
                 classification=False):
        self.classify = classification
        self.n_h = n_h
        self.embed = nn.Linear(x_dim, h_dim)
        self.convs = nn.ModuleList([CGConv(h_dim,edge_dim)
                                    for _ in range(n_conv)])
        self.pool = aggr.MeanAggregation()
        self.lin = nn.Linear(h_dim, hout_dim)
        self.softplus = nn.Softplus()
        if n_h > 1:
            self.lins = nn.ModuleList([nn.Linear(hout_dim, hout_dim) for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h-1)])
        if classification == True:
            self.out = nn.Linear(hout_dim, 2)
            self.dropout = nn.Dropout()
            self.lsm = nn.LogSoftmax(dim=1)
        else:
            self.out = nn.Linear(hout_dim, 1)
    
    def forward(self, data):
        x = self.embed(data.x)
        for cgconv in self.convs:
            x = cgconv(x, data.edge_index, data.edge_attr)
        x = self.pool(x, index=data.batch)
        x = self.lin(x)
        x = self.softplus(x)
        if self.classify == True:
            x = self.dropout(x)
        if self.n_h > 1:
            for lin, sp in zip(self.lins,self.softpluses):
                x = lin(x)
                x = sp(x)
        out = self.out(x)
        if self.classify == True:
            out = self.lsm(out)
            
        return out
            




