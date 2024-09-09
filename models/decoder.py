import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    """
    Decoder layer that integrates self-attention, cross-attention, graph construction, RNN, and convolution layers.
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, seq_length=96, dropout=0.1, activation="relu", device=None, tanhalpha=3, subgraph_size=20, node_dim=40, propalpha=0.05, skip_channels=64, end_channels=128, gcn_depth=2):
        """
        Initializes the DecoderLayer with attention, graph construction, RNN, and convolution layers.

        Args:
            self_attention (nn.Module): Self-attention module.
            cross_attention (nn.Module): Cross-attention module.
            d_model (int): Dimension of input features.
            d_ff (int, optional): Dimension of feed-forward layers. Defaults to 4 * d_model.
            seq_length (int, optional): Sequence length. Defaults to 96.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            activation (str, optional): Activation function ('relu' or 'gelu'). Defaults to 'relu'.
            device (torch.device, optional): Device for computation. Defaults to None.
            tanhalpha (float, optional): Scaling factor for tanh activation in graph construction. Defaults to 3.
            subgraph_size (int, optional): Size of subgraphs. Defaults to 20.
            node_dim (int, optional): Node embedding dimension for graph construction. Defaults to 40.
            propalpha (float, optional): Weighting factor for graph propagation. Defaults to 0.05.
            gcn_depth (int, optional): Depth of graph convolution layers. Defaults to 2.
        """
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.num_nodes = d_ff
        self.residual_channels = d_model
        self.idx = torch.arange(self.num_nodes).to(device)

        # Graph construction
        self.gc = GraphConstructor(self.num_nodes, subgraph_size, node_dim, device=device, alpha=tanhalpha)

        # Attention mechanisms
        self.self_attention = self_attention
        self.cross_attention = cross_attention

        # Convolution and RNN layers
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.rnnLayer = RnnLayer(in_channels=d_ff, hid_size=d_model, num_layers=5)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # Feed-forward and normalization layers
        self.fc = nn.Linear(in_features=d_model, out_features=d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # Graph convolution layers for graph-based propagation
        self.gconv1 = MixProp(seq_length, d_model, gcn_depth, dropout, propalpha)
        self.gconv2 = MixProp(seq_length, d_model, gcn_depth, dropout, propalpha)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Forward pass for the decoder layer.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_length, d_model].
            cross (torch.Tensor): Cross input tensor for cross-attention.
            x_mask (torch.Tensor, optional): Mask for self-attention.
            cross_mask (torch.Tensor, optional): Mask for cross-attention.

        Returns:
            torch.Tensor: Output after processing through self-attention, cross-attention, RNN, and graph-based layers.
        """
        # Self-attention
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        # Cross-attention
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x = self.norm2(x)

        # Convolution and RNN layers
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        hc = self.rnnLayer(y.transpose(-1, 0).transpose(-1, 1))
        hc = self.fc(hc).transpose(1, 0).transpose(-1, 1)
        y = y + hc

        # Gating mechanism
        filter = torch.tanh(y)
        gate = torch.sigmoid(y)
        y2 = filter * gate

        # Graph construction
        adp = self.gc(self.idx)

        # Graph convolution operations
        gx = self.gconv1(hc, adp) + self.gconv2(hc, adp.transpose(1, 0))

        # Final convolution and normalization
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y2 = self.dropout(self.conv2(y2).transpose(-1, 1))

        return self.norm3(x + y2 + gx.transpose(-1, 1))

class GraphConstructor(nn.Module):
    """
    A class used to dynamically construct graph adjacency matrices based on node features or embeddings.
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

    def forward(self, idx):
        """
        Forward pass to compute the adjacency matrix.

        Args:
            idx (torch.Tensor): Tensor containing node indices.

        Returns:
            torch.Tensor: Sparse adjacency matrix of the graph.
        """
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))

        # Sparsification: keep only top-k values
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        
        return adj

class MixProp(nn.Module):
    """
    Mixture of propagation layers for graph convolution networks (GCNs).
    """
    def __init__(self, seq_length, c_out, gdep, dropout, alpha):
        super(MixProp, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear((gdep + 1) * seq_length, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        """
        Forward pass for the MixProp layer.

        Args:
            x (torch.Tensor): Input features [batch_size, num_nodes, feature_dim].
            adj (torch.Tensor): Adjacency matrix [num_nodes, num_nodes].

        Returns:
            torch.Tensor: Output after applying multiple graph convolutions and MLP.
        """
        adj = adj + torch.eye(adj.size(0)).to(x.device)  # Add self-loops
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)  # Normalize adjacency matrix by degree

        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)

        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)

        return ho

class NConv(nn.Module):
    """
    A graph convolutional layer that performs node feature aggregation using the adjacency matrix.
    """
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, A):
        """
        Forward pass for the graph convolution operation.

        Args:
            x (torch.Tensor): Input features [batch_size, num_nodes, feature_dim].
            A (torch.Tensor): Adjacency matrix [num_nodes, num_nodes].

        Returns:
            torch.Tensor: Output after applying graph convolution.
        """
        x = torch.einsum('nwc,wv->nvc', (x, A))
        return x.contiguous()

class Linear(nn.Module):
    """
    A simple linear layer using 1x1 convolution for feature transformation.
    """
    def __init__(self, c_in, c_out, bias=True):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        """
        Forward pass for the linear transformation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed output.
        """
        return self.mlp(x.transpose(0, 1)).transpose(0, 1)

class RnnLayer(nn.Module):
    """
    LSTM-based recurrent neural network layer.
    """
    def __init__(self, in_channels, hid_size=128, num_layers=5, dropout=0.1):
        super(RnnLayer, self).__init__()
        self.hidden_size = hid_size
        self.lstm = nn.LSTM(in_channels, hidden_size=hid_size, num_layers=num_layers, dropout=dropout)

    def forward(self, x):
        """
        Forward pass for LSTM layer.

        Args:
            x (torch.Tensor): Input tensor of shape [sequence_length, batch_size, feature_dim].

        Returns:
            torch.Tensor: Output from the LSTM layer.
        """
        hc = self.lstm(x)
        return hc[0]
    
class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x