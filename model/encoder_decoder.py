import torch
import torch.nn as nn
from model.modules import SA_GC

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
    def __init__(self, input_dim, hidden_dims, n_layers, A, device):
        super(SAGC_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.H, self.C = [],[]
        self.num_joint = A.shape[1]
        self.device = device

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
                self.H[j], self.C[j] = cell(input_, (self.H[j],self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1],(self.H[j],self.C[j]))

        return (self.H,self.C) , self.H   # (hidden, output)

    def initHidden(self,batch_size):
        self.H, self.C = [],[]
        for i in range(self.n_layers):
            self.H.append(torch.zeros(batch_size, self.hidden_dims[i], 1, self.num_joint).to(self.device))
            self.C.append(torch.zeros(batch_size, self.hidden_dims[i], 1, self.num_joint).to(self.device))

    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C


class Encoder_z0_RNN(nn.Module):
    def __init__(self, latent_dim, A, device = torch.device("cpu")):

        super(Encoder_z0_RNN, self).__init__()

        self.latent_dim = latent_dim
        self.device = device

        self.hiddens_to_z0 = nn.Sequential(
            SA_GC(self.latent_dim, self.latent_dim, A),
            SA_GC(self.latent_dim, 2*self.latent_dim, A),
        )

        self.sagc_lstm =  SAGC_LSTM(
            input_dim=self.latent_dim, hidden_dims=[128, 128, self.latent_dim],
            n_layers=3, A=A, device=device
        )


    def forward(self, data, run_backwards=True):
        # data : B C T V

        B, C, T, V = data.shape

        time_set = list(range(T))

        if run_backwards:
            time_set = time_set[::-1]

        assert(not torch.isnan(data).any())
        for i, idx in enumerate(time_set):
            _, output = self.sagc_lstm(data[:,:,idx:idx+1,:], (i==0))
        # LSTM output shape: (seq_len, batch, num_directions * hidden_size)

        z0 = self.hiddens_to_z0(output[-1])
        mean, std = torch.split(z0, C, dim=1)
        std = std.abs()
        assert(not torch.isnan(mean).any())
        assert(not torch.isnan(std).any())

        return mean, std
