import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size=1024, latent_dim=8):
        super(VAE, self).__init__()
        # Layer sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc21 = nn.Linear(self.hidden_size, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_size, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, z
    
    def freeze_encoder(self):
        for param in list(self.fc1.parameters()) + list(self.fc21.parameters()) + list(self.fc22.parameters()):
            param.requires_grad = False
