import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pvae.utils import Constants
from pvae.manifolds import PoincareBall, Euclidean
# TODO: 여러 개의 csv 파일을 다루는 것.

class AVATARDataset(Dataset):
    def __init__(self, avatar_csv_path, subsequence_length=300, overlap_length=60):
        """
        It only reads one csv file.

        Args:
            avatar_csv_path (_type_): _description_
        """
        self.avatar_csv_path = avatar_csv_path
        if not os.path.exists(self.avatar_csv_path):
            raise ValueError("CSV file does not exist.")

        self.csv_data = np.genfromtxt(self.avatar_csv_path, delimiter=',')
        self.data = self.cut_sequences(subsequence_length, overlap_length)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def cut_sequences(self, subsequence_length, overlap_length):
        """
        Cut the sequences into subsequences.

        Args:
            subsequence_length (int): Length of the subsequence.
            overlap_length (int): Length of the overlap between two consecutive subsequences.

        Returns:
            list: List of subsequences.
        """
        # Get the number of subsequences.
        
        num_subsequences = int((self.csv_data.shape[0] - overlap_length) / (subsequence_length - overlap_length))
        # Cut the sequences into subsequences.
        subsequences = []
        for i in range(num_subsequences):
            subsequence = self.csv_data[i*(subsequence_length-overlap_length):i*(subsequence_length-overlap_length)+subsequence_length, :]
            subsequences.append(subsequence)

        return subsequences
    

class AVATARpVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers, architecture="GRU"):
        super(AVATARpVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if architecture == "GRU":
            RNN = nn.GRU
        elif architecture == "LSTM":
            RNN = nn.LSTM
        
        # Encoder

# Isotropic Gaussian prior

class pEncoderLinear(nn.Module):
    # It maps the input to the Euclidean manifold.
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers, manifold, architecture="GRU", bidirectional=True):
        super(pEncoderLinear, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if architecture == "GRU":
            RNN = nn.GRU
        elif architecture == "LSTM":
            RNN = nn.LSTM
        
        self.hidden_state_dim = hidden_dim if not bidirectional else (2*hidden_dim)
        self.nonlinearity = nn.ReLU()
        self.rnn = RNN(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.l1 = nn.Linear(self.hidden_state_dim, self.hidden_state_dim)

        self.l21 = nn.Linear(self.hidden_state_dim, latent_dim) # frechet mean, manifold dimensionality
        self.l22 = nn.Linear(self.hidden_state_dim, 1) # variance

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, latent_dim).
        """
        # x: (batch_size, seq_len, input_dim)
        e = self.nonlinearity(self.rnn(x)[0]) # e: (batch_size, seq_len, hidden_state_dim)
        e = self.nonlinearity(self.l1(e))
        mu = self.l21(e) # mu: (batch_size, seq_len, latent_dim)
        var = F.softplus(self.l22(e)) + Constants.eta # var: (batch_size, seq_len, 1)

        return mu, var
    
class pDecoderLinear(nn.Module):
    # It maps a latent vector from the Euclidean manifold to the input space.
    def __init__(self, input_dim, latent_dim, hidden_dim, manifold, num_layers, architecture="GRU", bidirectional=True):
        super(pDecoderLinear, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if architecture == "GRU":
            RNN = nn.GRU
        elif architecture == "LSTM":
            RNN = nn.LSTM

        self.nonlinearity = nn.ReLU()
        self.hidden_state_dim = hidden_dim if not bidirectional else (2*hidden_dim)
        self.rnn = RNN(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.l1 = nn.Linear(self.hidden_state_dim, self.hidden_state_dim)
        self.l2 = nn.Linear(self.hidden_state_dim, input_dim)

    def forward(self, z):
        """
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, seq_len, latent_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, input_dim).
        """
        # z: (batch_size, seq_len, latent_dim)
        d = self.nonlinearity(self.rnn(z)[0])
        d = self.nonlinearity(self.l1(d))
        mu = self.l2(d) # mu: (batch_size, seq_len, input_dim)
        

        return mu


class pEncoderWrapped(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, manifold, num_layers, architecture="GRU", bidirectional=True):
        super(pEncoderWrapped, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.manifold = manifold
        if architecture == "GRU":
            RNN = nn.GRU
        elif architecture == "LSTM":
            RNN = nn.LSTM

        self.nonlinearity = nn.ReLU()
        self.hidden_state_dim = hidden_dim if not bidirectional else (2*hidden_dim)
        self.rnn = RNN(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional, )
        self.l1 = nn.Linear(self.hidden_state_dim, self.hidden_state_dim)
        self.l21 = nn.Linear(self.hidden_state_dim, manifold.coord_dim) # frechet mean, manifold dimensionality
        self.l22 = nn.Linear(self.hidden_state_dim, 1) # variance TODO: non-isotropic Gaussian prior

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, latent_dim).
        """
        # x: (batch_size, seq_len, input_dim)
        e = self.nonlinearity(self.rnn(x)[0]) # e: (batch_size, seq_len, hidden_dim)
        e = self.l1(e)
        mu = self.l21(e) # mu: (batch_size, seq_len, latent_dim)
        mu = self.manifold.expmap0(mu)
        var = F.softplus(self.l22(e)) + Constants.eta # var: (batch_size, seq_len, 1)

        return mu, var
    
class pDecoderWrapped(nn.Module):
    def __init__(self, latent_dim, hidden_dim, manifold, num_layers, data_shape, architecture="GRU", bidirectional=True):
        """
        Decode latent vector from the latent manifold to the input space.

        Args:
            latent_dim (_type_): The dimensionality of the latent manifold.
            hidden_dim (_type_): The dimensionality of the hidden vectors
            manifold (_type_): The latent manifold. 
            num_layers (_type_): The number of layers of the RNN.
            data_shape (_type_): The shape of input data. (seq_len, input_dim)
            architecture (str, optional): The RNN architecture. 'GRU' or "LSTM". Defaults to "GRU".
            bidirectional (bool, optional): If True, it uses bidirectional RNN. Else, it doesn't. Defaults to True.
        """


        super(pDecoderLinear, self).__init__()
        
        self.data_shape = data_shape
        self.input_dim = data_shape[1]
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.manifold = manifold
        if architecture == "GRU":
            RNN = nn.GRU
        elif architecture == "LSTM":
            RNN = nn.LSTM

        self.nonlinearity = nn.ReLU()
        self.hidden_state_dim = hidden_dim if not bidirectional else (2*hidden_dim)
        self.rnn = RNN(self.manifold.coord_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional, )
        self.l1 = nn.Linear(self.hidden_state_dim, self.hidden_state_dim)
        self.l2 = nn.Linear(self.hidden_state_dim, np.prod(data_shape)) 
        

    def forward(self, z):
        """
        maps a latent vector from the latent manifold to the input space.

        Args:
            z (_type_): laten vector with shape (batch_size, seq_len, latent_dim).

        Returns:
            _type_: output tensor with shape (batch_size, seq_len, input_dim) in the input space.
        """
        z = self.manifold.logmap0(z)
        e = self.nonlinearity(self.rnn(z)[0]) # e: (batch_size, seq_len, hidden_dim)
        e = self.nonlinearity(self.l1(e))
        mu = self.l2(e) # mu: (batch_size, seq_len*input_dim)

        return mu.view(z.size()[0], *self.data_shape) # mu: (batch_size, seq_len, input_dim)
    

def train(dataset, batch_size, num_workers, shuffle):
    """
    Args:
        dataset (Dataset): Dataset object.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: DataLoader object.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)



if __name__ == "__main__":
    csv_path = "/Users/youngmin/Documents/research/behavit/pvae/data/Behav_Dataset/dataset/y3a6/coords/adult_8294.csv"
    dataset = AVATARDataset(csv_path)
    print(dataset[0].shape)

    train(dataset, 32, 0, True)