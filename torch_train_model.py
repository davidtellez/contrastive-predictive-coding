
import torch
from torch import nn
from torch import functional as F
from torch import optim
from torch.autograd import Variable


cuda = torch.cuda.is_available()


class NetworkEncoder(nn.Module):
    """NetworkEncoder: torch.nn.Module

    Transforms a batch of images into an encoded representation with
    `code_size` elements."""

    def __init__(self, code_size, in_channels: int = 3):
        super().__init__()
        self.code_size = code_size
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.linear2 = nn.Linear(256, code_size)

    def set_num_dense(self, n):
        self._lin1 = nn.Linear(n, 256)
        self._leaky1 = nn.LeakyReLU()
        self._lin1 = self._lin1.cuda() if cuda else self._lin1

    def linear1(self, x):
        try:
            return self._leaky1(self._lin1(x))
        except AttributeError:
            self.set_num_dense(x.size(-1))
            return self.linear1(x)

    def dense(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return self.linear2(x)

    def forward(self, x):
        if len(x.shape) == 5:
            return torch.stack([self.forward(xi) for xi in x], dim=0)
        x = self.convolution(x)
        return self.dense(x)


class Autoregressive(nn.Module):
    """Autoregressive: torch.nn.Module

    Processes a series of encoded images to summarize their content in a
    context vector of length `hidden_size`
    """

    def __init__(self):
        super().__init__()
        self.hidden_size = 256

    def set_input_size(self, input_size):
        self._rnn = nn.GRU(input_size=input_size,
            hidden_size=self.hidden_size, num_layers=1)
        self._rnn = self._rnn.cuda() if cuda else self._rnn

    def rnn(self, x, hidden):
        assert len(x.shape) == 3, f"""
        x.shape should be (sequence, batch, features) {x.shape}"""
        try:
            return self._rnn(x, hidden)
        except AttributeError:
            self.set_input_size(x.size(2))
            return self.rnn(x, hidden)

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return h.cuda() if cuda else h

    def forward(self, x, hidden=None):
        """return last application of `rnn`

        x.shape : (sequence, batch, features)
        """
        assert (hidden is None) or len(hidden.shape) == 3
        hidden = hidden or self.init_hidden(x.size(1))
        x, hidden = self.rnn(x, hidden)
        return x[-1, :, :]


class Predictor(nn.Module):
    """Predictor: torch.nn.Module

    Predict a number future encodings given a context vector.
    """
    
    def __init__(self, code_size, predict_terms):
        super().__init__()
        self.predict_terms = predict_terms
        self.code_size = code_size

    def set_encoding_size(self, encoding_size):
        self._nets = nn.ModuleList([
            nn.Linear(encoding_size, code_size)
            for _ in range(self.predict_terms)
        ])
        self._nets = self._nets.cuda() if cuda else self._nets

    def nets(self, x):
        assert len(x.shape) == 2, f"""
        x.shape should be (batch, encoding) {x.shape}"""
        try:
            return torch.stack([net(x) for net in self._nets], dim=0)
        except AttributeError:
            self.set_encoding_size(x.size(-1))
            return self.nets(x)

    def forward(self, x):
        prediction = self.nets(x)
        return prediction


class CPCNetwork(nn.Module):
    """CPCNetwork: torch.nn.Module

    Encodes two series of images.  Produces a context from the first series and
    predicts elements in the second encoded series.
    """

    def __init__(self, code_size, predict_terms):
        super().__init__()
        self.encoder = NetworkEncoder(code_size, in_channels=3)
        self.ar = Autoregressive()
        self.predictor = Predictor(code_size, predict_terms)

    def forward(self, x, y):
        encoded_x = self.encoder(x)
        context = self.ar(encoded_x)
        prediction = self.predictor(context)
        logit = torch.mean(prediction * self.encoder(y), dim=(0, -1))
        return logit


if __name__ == "__main__":

    import numpy as np
    from data_utils import SortedNumberGenerator

    epochs = 10
    batch_size = 32
    code_size = 128
    lr = 1e-3
    terms = 4
    predict_terms = 4
    image_size = 64
    color = True

    # setup network and training
    cpcnet = CPCNetwork(code_size, predict_terms)
    cpcnet = cpcnet.cuda() if cuda else cpcnet
    bcelogit_fn = nn.BCEWithLogitsLoss(reduction='mean')
    sigmoid = nn.Sigmoid()
    opt = optim.Adam(cpcnet.parameters(), lr=lr)

    # setup data
    train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                       positive_samples=batch_size // 2, predict_terms=predict_terms,
                                       image_size=image_size, color=color, rescale=True)

    validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=batch_size // 2, predict_terms=predict_terms,
                                            image_size=image_size, color=color, rescale=True)

    def prepare(x, y, z):
        """return batch of data ready for network"""
        data = (
            Variable(torch.from_numpy(x.swapaxes(0, 1).swapaxes(2, 4))),
            Variable(torch.from_numpy(y.swapaxes(0, 1).swapaxes(2, 4))),
            Variable(torch.from_numpy(z))[:, 0]
        )
        return (d.cuda() for d in data) if cuda else data

    def accuracy(logit, target):
        """return accuracy from log(P) and targets"""
        preds = 0.5 < sigmoid(logit)
        acc = torch.sum(preds == target).float() / target.shape[0]
        return acc

    def evaluate(x, y, target):
        """return loss and accuracy of cpc-net predictions"""
        product = cpcnet(x, y)
        loss = bcelogit_fn(product, target.float())
        acc = accuracy(product, target.byte())
        return loss, acc

    for epoch in range(epochs):
        # training
        cpcnet.train()
        for n_batch, data in enumerate(train_data):
            opt.zero_grad()
            x, y, z = prepare(*data[0], data[1])
            loss, acc = evaluate(x, y, z)
            print(f"[batch {n_batch}] train loss = {loss.item():.4g}; acc = {acc.item():.4g}")
            loss.backward()
            opt.step()
        print()
        # validation
        cpcnet.eval()
        for n_batch, data in enumerate(validation_data):
            with torch.no_grad():
                x, y, z = prepare(*data[0], data[1])
                loss, acc = evaluate(x, y, z)
                print(f"[batch {n_batch}] val loss = {loss.item():.4g}; acc = {acc.item():.4g}")
