import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.optim import Adam


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        # 1x1 convolution for residual connection
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # residual connection
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i # dilated conv
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_channels, kernel_size=2, dropout=0):
        super().__init__()
        self.tcn = TemporalConvNet(in_dim, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], out_dim)

    def forward(self, x):
        x = self.tcn(x)
        x = self.linear(x[:, :, -1])
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    print("validating TCN performance")

    # parameters
    obs_dim = 11
    out_dim = 11
    # I suppose these are hidden layer sizes? so the obs dim is reduced to this size
    num_channels = [5, 7, 5] 
    kernel_size = 3

    # data
    batch_size = 50
    n_sample = 1000
    sequence_length = 17

    # lens = torch.randint(3, 7, (n_sample,))
    # xlst = []
    # y = torch.zeros((n_sample,), device=device)
    # for i in range(n_sample):
    #     xlst.append(torch.randn((obs_dim, lens[i]), device=device))
    #     #y[i] = xlst[-1][...,-1]
    
    # print(xlst)
    # x = torch.nested.nested_tensor(xlst)

    x = torch.rand(n_sample, obs_dim, sequence_length, device=device)
    y = x[...,-1].clone()

    # init model
    tcnnet = TCN(
        in_dim=obs_dim,
        out_dim=out_dim,
        num_channels=num_channels,
        kernel_size=kernel_size,
    )
    tcnnet = tcnnet.to(device=device)

    #print(tcnnet(x))
    
    # training
    print("training")
    optimizer = Adam(tcnnet.parameters(), lr=0.01, eps=1e-5)
    loss_fn = nn.MSELoss()
    
    # epochs
    for i in range(10):
        tcnnet.train()
        train_loss = 0.0
        for j in range(n_sample//batch_size):
            optimizer.zero_grad() # Reset gradients
            z = tcnnet(x[j*batch_size:(j+1)*batch_size]) # forwad pass
            J = loss_fn(z[...], y[j*batch_size:(j+1)*batch_size]) # calc loss
            J.backward() # backward pass
            optimizer.step() # update weights

            train_loss += (J.detach().item() - train_loss) / (j + 1)

        print(f"Epoch: {i}\t train loss: {train_loss}")

