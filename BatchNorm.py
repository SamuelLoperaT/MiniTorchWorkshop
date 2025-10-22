import torch
class BatchNorm1D:
    """
    Batch Normalization for 2D inputs: (batch, features).

    TRAIN: compute batch stats, normalize, update running stats, support backward().
    EVAL:  use running stats, no updates, typically no backward().
    """

    def __init__(self, n_features, eps=1e-5, momentum=0.1, device="cpu"):
        self.eps = eps
        self.momentum = momentum
        self.device = device

        # Learnable affine parameters
        self.gamma = torch.ones(n_features, device=device, requires_grad=False)
        self.beta  = torch.zeros(n_features, device=device, requires_grad=False)

        # Running (inference) statistics
        self.running_mean = torch.zeros(n_features, device=device, requires_grad=False)
        self.running_var  = torch.ones(n_features,  device=device, requires_grad=False)

        # Mode flag
        self.training = True

        # Caches for backward
        self.X = None
        self.X_hat = None
        self.batch_mean = None
        self.batch_var = None
        self.std = None

        # Grads for parameters
        self.dgamma = None
        self.dbeta  = None

    def train(self): 
        self.training = True
        return self
    
    def eval(self):  
        self.training = False
        return self

    def forward(self, X):
        """
        Args:
            X: (batch, features)
        Returns:
            Y: (batch, features)
        """
        if self.training:
            # batch stats
            self.batch_mean = X.mean(dim=0)
            self.batch_var  = X.var(dim=0, unbiased=False)

            # normalization
            self.std = torch.sqrt(self.batch_var + self.eps)
            self.X_hat = (X - self.batch_mean) / self.std

            # update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * self.batch_var

        else:
            # use running stats
            self.std = torch.sqrt(self.running_var + self.eps)
            self.X_hat = (X - self.running_mean) / self.std

        # cache input for backward
        self.X = X

        # affine transform
        Y = self.gamma * self.X_hat + self.beta
        return Y

    def backward(self, dY):
        """
        Args:
            dY: upstream gradient (batch, features)
        Returns:
            dX: gradient wrt input X (batch, features)
        """
        if not self.training:
            raise RuntimeError("Backward called in eval() mode. Use training mode for gradient computation.")

        m = dY.size(0)  # batch size

        # parameter gradients
        self.dbeta  = dY.sum(dim=0)
        self.dgamma = (dY * self.X_hat).sum(dim=0)

        # gradient wrt normalized activations
        dx_hat = dY * self.gamma

        x_mu   = self.X - self.batch_mean
        invstd = 1.0 / self.std

        # grads for variance and mean
        dvar  = torch.sum(dx_hat * x_mu * -0.5 * (invstd ** 3), dim=0)
        dmean = torch.sum(-dx_hat * invstd, dim=0) + dvar * torch.mean(-2.0 * x_mu, dim=0)

        # final gradient wrt input
        dX = dx_hat * invstd + (2.0 / m) * x_mu * dvar + dmean / m
        return dX

    def update(self, lr):
        """
        Simple SGD update for gamma and beta.
        """
        self.gamma -= lr * self.dgamma
        self.beta  -= lr * self.dbeta