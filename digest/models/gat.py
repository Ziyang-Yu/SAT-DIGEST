from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv

from digest.models import ScalableGNN


class GAT(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 hidden_heads: int, out_channels: int, out_heads: int,
                 num_layers: int, dropout: float = 0.0, linear: bool = False,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels * hidden_heads, num_layers,
                         pool_size, buffer_size, device)

        self.in_channels = in_channels
        self.hidden_heads = hidden_heads
        self.out_channels = out_channels
        self.out_heads = out_heads
        self.dropout = dropout

        self.convs = ModuleList()
        for i in range(num_layers - 1):
            in_dim = in_channels if i == 0 else hidden_channels * hidden_heads
            conv = GATConv(in_dim, hidden_channels, hidden_heads, concat=True,
                           dropout=dropout, add_self_loops=False)
            self.convs.append(conv)

        conv = GATConv(hidden_channels * hidden_heads, out_channels, out_heads,
                       concat=False, dropout=dropout, add_self_loops=False)
        self.convs.append(conv)

        self.reg_modules = self.convs
        self.nonreg_modules = ModuleList()

    def reset_parameters(self):
        super().reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, batch_size, his_data, 
                subgraph_adj, auxiliary_model, hist_embs, rank, *args) -> Tensor:
        local_push_data = []
        corrected_embs = []
        device = torch.device("cuda:"+str(rank))
        print(self.histories)
        for i, conv in enumerate(zip(self.convs[:-1])):
            conv = conv[0]
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = conv((x, x[:adj_t.size(0)]), adj_t)
            x = F.elu(x)
            # x = self.push_and_pull(history, x, *args)
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
        x = self.convs[-1]((x, x[:adj_t.size(0)]), adj_t)
        return x, local_push_data, corrected_embs

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[layer]((x, x[:adj_t.size(0)]), adj_t)

        if layer < self.num_layers - 1:
            # x = x.elu()
            x = F.elu(x)
        return x
