import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        return grads.neg() * ctx.lambda_, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(input_size=32 * 5, hidden_size=64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
        feats = self.feature_extractor(x)
        feats = feats.reshape(N, T, -1)
        out, (hn, cn) = self.lstm(feats)
        z = hn[-1]
        y = self.classifier(z)
        return y, z

class DANN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(input_size=32 * 5, hidden_size=64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(64, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x, alpha=1.0):
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
        feats = self.feature_extractor(x)
        feats = feats.reshape(N, T, -1)
        out, (hn, cn) = self.lstm(feats)
        z = hn[-1]
        y = self.classifier(z)
        
        rev_z = GradientReversal.apply(z, alpha)
        d = self.domain_classifier(rev_z)
        return y, d, z
