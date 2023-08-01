from typing import Optional, Tuple, Any, List

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv

from digest.models import ScalableGNN




class GCN(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 drop_input: bool = True, batch_norm: bool = False,
                 residual: bool = False, linear: bool = False,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual
        self.linear = linear

        self.lins = ModuleList()
        if linear:
            self.lins.append(Linear(in_channels, hidden_channels))
            self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0 and not linear:
                in_dim = in_channels
            if i == num_layers - 1 and not linear:
                out_dim = out_channels
            conv = GCNConv(in_dim, out_dim, normalize=False)
            self.convs.append(conv)

        self.bns = ModuleList()
        for i in range(num_layers):
            bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)
        

    @property
    def reg_modules(self):
        if self.linear:
            return ModuleList(list(self.convs) + list(self.bns))
        else:
            return ModuleList(list(self.convs[:-1]) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.lins if self.linear else self.convs[-1:]

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    # def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
    def forward(self, x: Tensor, adj_t: SparseTensor, batch_size, his_data, 
                subgraph_adj, auxiliary_model, hist_embs, rank, *args) -> Tuple[Any, List[Any]]:

        device = torch.device("cuda:"+str(rank))
        local_push_data = []
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.linear:
            x = self.lins[0](x).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        # for conv, bn, hist in zip(self.convs[:-1], self.bns, self.histories):
        corrected_embs = []
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            h = conv(x, adj_t)
            if self.batch_norm:
                h = bn(h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]

            x = h.relu_()
            # x = self.push_and_pull(hist, x, *args)
            # x = torch.cat([x[:batch_size], his_data[i].to(f'cuda:{x.get_device()}')], dim=0)
            #print("previous", x.size())

            """
            forward propagation of auxiliary model
            """

            while len(hist_embs[i].pull()) > 3:
                hist_embs[i].pop(0)
            #print(len(hist_embs[i].pull()))
            hist_embs[i].push(his_data[i].clone().cpu())
            zero_tensor = torch.zeros([batch_size, x.size()[1]],dtype=torch.float).to(device)
            pull_data_aux = hist_embs[i].pull()
            input_aux = []
            for i in range(len(pull_data_aux)):
                input_aux.append(torch.cat([zero_tensor, pull_data_aux[i].clone().to(device)]))
            input_aux = torch.stack(input_aux).detach().to(device)
            auxiliary_model = auxiliary_model.to(device)
            subgraph_adj = subgraph_adj.to(device)
            output_aux = auxiliary_model(input_aux, subgraph_adj)
            corrected_embs.append(output_aux[batch_size:])

            
            

            x = torch.cat([x[:batch_size], output_aux.clone().detach()[batch_size:]], dim=0)

            local_push_data.append(x[:batch_size])

            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[-1](x, adj_t)

        if not self.linear:
            return h, local_push_data, corrected_embs

        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        h = h.relu_()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lins[1](h), local_push_data, corrected_embs

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        if layer == 0:
            if self.drop_input:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.linear:
                x = self.lins[0](x).relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[layer](x, adj_t)

        if layer < self.num_layers - 1 or self.linear:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = h.relu_()

        if self.linear:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.lins[1](h)

        return h
