import torch
import torch.nn as nn
from model.modules import SA_GC, GCN


class SAGC_LSTM_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, A):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(SAGC_LSTM_Cell, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.sagc = SA_GC(input_dim + hidden_dim, 4 * hidden_dim, A)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden): # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.sagc(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class SAGC_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_layers, A):
        super(SAGC_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.H, self.C = [],[]
        self.num_joint = A.shape[1]

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            print('layer ',i,'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(SAGC_LSTM_Cell(cur_input_dim, self.hidden_dims[i], A))
        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size) # init Hidden at each forward start

        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j].to(input_.device),self.C[j].to(input_.device)))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1].to(input_.device),(self.H[j].to(input_.device),self.C[j].to(input_.device)))

        return (self.H,self.C) , self.H   # (hidden, output)

    def initHidden(self,batch_size):
        self.H, self.C = [],[]
        for i in range(self.n_layers):
            self.H.append(torch.zeros(batch_size, self.hidden_dims[i], 1, self.num_joint))
            self.C.append(torch.zeros(batch_size, self.hidden_dims[i], 1, self.num_joint))

    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C


class GCN_LSTM_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, A):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(GCN_LSTM_Cell, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.sagc = GCN(input_dim + hidden_dim, 4 * hidden_dim, A)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden): # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.sagc(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class GCN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_layers, A):
        super(GCN_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.H, self.C = [],[]
        self.num_joint = A.shape[1]

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            print('layer ',i,'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(GCN_LSTM_Cell(cur_input_dim, self.hidden_dims[i], A))
        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size) # init Hidden at each forward start

        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j].to(input_.device),self.C[j].to(input_.device)))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1].to(input_.device),(self.H[j].to(input_.device),self.C[j].to(input_.device)))

        return (self.H,self.C) , self.H   # (hidden, output)

    def initHidden(self,batch_size):
        self.H, self.C = [],[]
        for i in range(self.n_layers):
            self.H.append(torch.zeros(batch_size, self.hidden_dims[i], 1, self.num_joint))
            self.C.append(torch.zeros(batch_size, self.hidden_dims[i], 1, self.num_joint))

    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C


class SAGC_LSTM_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, A):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(SAGC_LSTM_Cell, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.sagc = SA_GC(input_dim + hidden_dim, 4 * hidden_dim, A)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden): # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.sagc(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class Encoder_z0_RNN(nn.Module):
    def __init__(self, latent_dim, A, device = torch.device("cpu")):

        super(Encoder_z0_RNN, self).__init__()

        self.latent_dim = latent_dim
        self.device = device

        self.hiddens_to_z0 = nn.Sequential(
            GCN(self.latent_dim, self.latent_dim, A),
            GCN(self.latent_dim, self.latent_dim, A),
        )

        self.sagc_lstm =  GCN_LSTM(
            input_dim=self.latent_dim, hidden_dims=[self.latent_dim, self.latent_dim],
            n_layers=2, A=A
        )


    def forward(self, data, run_backwards=False):
        # data : B C T V

        B, C, T, V = data.shape

        time_set = list(range(T))
        zs = []
        if run_backwards:
            time_set = time_set[::-1]

        assert(not torch.isnan(data).any())
        for i, idx in enumerate(time_set):
            _, output = self.sagc_lstm(data[:,:,idx:idx+1,:], (i==0))
            zs.append(output[-1])
        # LSTM output shape: (seq_len, batch, num_directions * hidden_size)

        z0 = torch.cat(zs, dim=2)
        z0 = self.hiddens_to_z0(z0)
        return z0

class RNN(nn.Module):
    def __init__(self, latent_dim, A, n_step):

        super(RNN, self).__init__()

        self.latent_dim = latent_dim

        self.N = n_step

        self.sagc_lstm =  GCN_LSTM(
            input_dim=self.latent_dim, hidden_dims=[self.latent_dim, self.latent_dim],
            n_layers=2, A=A
        )

        self.hiddens_to_z0 = nn.Sequential(
            GCN(self.latent_dim, self.latent_dim, A),
            GCN(self.latent_dim, self.latent_dim, A),
        )


    def forward(self, z0, time):
        # data : B C T V

        BT, C, V = z0.shape
        z0 = z0.unsqueeze(2)

        time_set = list(range(self.N))
        zs = [z0]
        assert(not torch.isnan(z0).any())
        for i, idx in enumerate(time_set):
            _, output = self.sagc_lstm(z0, (i==0))
            zs.append(output[-1])
        # LSTM output shape: (seq_len, batch, num_directions * hidden_size)
        zs = torch.stack(zs, dim=0)
        zs = zs.squeeze(3)
        z0 = self.hiddens_to_z0(z0)
        return zs
