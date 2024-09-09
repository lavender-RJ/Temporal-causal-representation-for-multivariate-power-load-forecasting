import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConstructor(nn.Module):
    """
    A class used to construct graph adjacency matrices dynamically.
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        """
        Initializes the graph constructor.
        
        Args:
            nnodes (int): Number of nodes in the graph.
            k (int): Parameter to control sparsity.
            dim (int): Dimension of embeddings or features.
            device (torch.device): Device for tensors.
            alpha (float, optional): Scaling factor for non-linearity. Defaults to 3.
            static_feat (torch.Tensor, optional): Static features of the graph nodes. Defaults to None.
        """
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        
        # If static features are provided, use them; otherwise, use embeddings
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

    def forward(self, idx, d):
        """
        Forward pass to compute the adjacency matrix.
        
        Args:
            idx (torch.Tensor): Indices of nodes.
            d (float): Threshold value for adjacency.

        Returns:
            adj (torch.Tensor): The adjacency matrix.
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
        adj = torch.sigmoid(self.alpha * a)
        adj = torch.where(adj < d, torch.tensor(0.0).to(self.device), adj)
        
        return adj

class RnnLayer(nn.Module):
    """
    RNN-based layer using LSTM.
    """
    def __init__(self, in_channels, hid_size=128, num_layers=5, dropout=0.1):
        """
        Initializes the RnnLayer.
        
        Args:
            in_channels (int): Input feature dimension.
            hid_size (int): Hidden state size of LSTM. Defaults to 128.
            num_layers (int): Number of LSTM layers. Defaults to 5.
            dropout (float): Dropout rate. Defaults to 0.1.
        """
        super(RnnLayer, self).__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size=hid_size, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

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

class NConv(nn.Module):
    """
    A graph convolution layer using matrix multiplication.
    """
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, A):
        """
        Forward pass for the graph convolution operation.
        
        Args:
            x (torch.Tensor): Node features of shape [batch_size, num_nodes, feature_dim].
            A (torch.Tensor): Adjacency matrix of shape [num_nodes, num_nodes].

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
        """
        Initializes the Linear layer.
        
        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            bias (bool): Whether to include a bias term. Defaults to True.
        """
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self, x):
        """
        Forward pass for the linear layer.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed output.
        """
        return self.mlp(x.transpose(0, 1)).transpose(0, 1)

class MixProp(nn.Module):
    """
    Mixture of propagation layers for GCNs.
    """
    def __init__(self, seq_length, c_out, gdep, dropout, alpha):
        """
        Initializes the MixProp layer.
        
        Args:
            seq_length (int): Length of the sequence.
            c_out (int): Output feature dimension.
            gdep (int): Number of graph convolutions.
            dropout (float): Dropout rate.
            alpha (float): Weighting factor for residual connections.
        """
        super(MixProp, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear((gdep+1) * seq_length, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        """
        Forward pass for the MixProp layer.
        
        Args:
            x (torch.Tensor): Input features of shape [batch_size, num_nodes, feature_dim].
            adj (torch.Tensor): Adjacency matrix of shape [num_nodes, num_nodes].

        Returns:
            torch.Tensor: Output after applying mixed propagation and MLP.
        """
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho
    
class EncoderLayer(nn.Module):
    """
    Encoder layer combining attention mechanism, convolution, and graph-based operations.
    """
    def __init__(self, attention, d_model, d_ff=None, seq_length=96, dropout=0.1, activation="relu", device=None, tanhalpha=3, subgraph_size=5.0, node_dim=10, propalpha=0.3, skip_channels=64, end_channels=128, gcn_depth=3):
        """
        Initializes the EncoderLayer.

        Args:
            attention (nn.Module): Attention mechanism module.
            d_model (int): Input and output feature dimension.
            d_ff (int, optional): Feed-forward layer dimension. Defaults to 4 * d_model.
            seq_length (int, optional): Sequence length. Defaults to 96.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            activation (str, optional): Activation function ('relu' or 'gelu'). Defaults to 'relu'.
            device (torch.device, optional): Device for tensor operations.
            tanhalpha (float, optional): Scaling factor for tanh activation. Defaults to 3.
            subgraph_size (float, optional): Size of subgraphs. Defaults to 5.0.
            node_dim (int, optional): Dimension of node embeddings. Defaults to 10.
            propalpha (float, optional): Propagation alpha for GCNs. Defaults to 0.3.
            skip_channels (int, optional): Number of skip connection channels. Defaults to 64.
            end_channels (int, optional): Number of output channels. Defaults to 128.
            gcn_depth (int, optional): Depth of GCN layers. Defaults to 3.
        """
        super(EncoderLayer, self).__init__()

        self.num_nodes = d_ff
        self.residual_channels = d_model
        self.idx = torch.arange(self.num_nodes).to(device)

        # Graph construction layer
        self.gc = GraphConstructor(self.num_nodes, 8, node_dim, device=device, alpha=tanhalpha, static_feat=None)
        
        d_ff = d_ff or 4 * d_model  # If d_ff is not provided, default to 4 * d_model
        self.attention = attention
        
        # Convolution layers
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.rnnLayer = RnnLayer(in_channels=d_ff, hid_size=d_model, num_layers=6)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # Normalization and activation
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(in_features=d_model, out_features=d_ff)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # Graph convolution layers and normalization
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.d = nn.Parameter(torch.tensor(0.69, requires_grad=True))
        self.gconv1.append(MixProp(self.num_nodes, self.residual_channels, gcn_depth, dropout, propalpha))
        self.gconv2.append(MixProp(self.num_nodes, self.residual_channels, gcn_depth, dropout, propalpha))

    def forward(self, x, attn_mask=None):
        """
        Forward pass for the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output after attention, convolution, and graph-based operations.
            torch.Tensor: Attention scores.
            torch.Tensor: Adjusted sum of graph adjacency matrix.
            torch.Tensor: Graph adjacency matrix.
        """
        # Apply attention mechanism
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = self.norm1(x)

        # Convolution and RNN layers
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y1 = y.transpose(-1, 0).transpose(-1, 1)
        hc = self.rnnLayer(y1)
        hc = self.fc(hc)
        hc = hc.transpose(1, 0).transpose(-1, 1)
        y = y + hc

        # Gating mechanism
        filter = torch.tanh(y)
        gate = torch.sigmoid(y)
        y2 = filter * gate

        # Dynamic graph construction
        adp = self.gc(self.idx, 0.79)
        gc_adjusted_sum = torch.sum(adp) - self.num_nodes

        # Apply final convolution and graph convolution
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y2 = self.dropout(self.conv2(y2).transpose(-1, 1))
        gx = self.gconv1[0](hc, adp) + self.gconv2[0](hc, adp.transpose(1, 0))

        return self.norm2(y2) + gx.transpose(-1, 1), attn, torch.abs(gc_adjusted_sum), adp

class Encoder(nn.Module):
    """
    Encoder module with attention layers and optional convolution layers.
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        Initializes the Encoder module.

        Args:
            attn_layers (list): List of attention layers.
            conv_layers (list, optional): List of convolution layers. Defaults to None.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        """
        Forward pass for the Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Final output from the encoder.
            list: List of attention maps from each layer.
            torch.Tensor: Adjusted sum of graph adjacency matrix.
            torch.Tensor: Graph adjacency matrix.
        """
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn, gc_adjusted_sum, adp = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn, gc_adjusted_sum, adp = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn, gc_adjusted_sum, adp = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, torch.abs(gc_adjusted_sum), adp

class EncoderStack(nn.Module):
    """
    Encoder stack module that allows stacking multiple encoders with varying sequence lengths.
    """
    def __init__(self, encoders, inp_lens):
        """
        Initializes the EncoderStack.

        Args:
            encoders (list): List of encoders.
            inp_lens (list): List of input lengths for each encoder.
        """
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        """
        Forward pass for the EncoderStack.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Stacked encoder outputs.
            list: List of attention maps from each encoder.
        """
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns