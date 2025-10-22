import torch

class Dropout:
    """
    Inverted Dropout (for fully-connected tensors [batch, features]).

    - TRAIN: randomly zeroes activations with prob p, and rescales by 1/(1-p)
             so the expected activation stays constant.
    - EVAL:  identity (no dropout, no scaling).
    """

    def __init__(self, p=0.5, device="cpu"):
        assert 0.0 <= p < 1.0, "p must be in [0, 1)."
        self.p = p
        self.device = device
        self.training = True
        self.mask = None  # cache for backward

    # Mode control
    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, X):
        """
        Forward pass of dropout.

        Args:
            X: Tensor of shape (batch, features)
        Returns:
            Tensor of same shape
        """
        if self.training and self.p > 0.0:
            keep_prob = 1.0 - self.p
            # Bernoulli mask {0,1} with prob=keep_prob
            self.mask = (torch.rand_like(X, device=self.device) < keep_prob).float()
            # Inverted dropout scaling
            self.mask /= keep_prob
            return X * self.mask
        else:
            # no dropout in eval mode
            self.mask = torch.ones_like(X, device=self.device)
            return X

    def backward(self, dY):
        """
        Backward pass of dropout.

        Args:
            dY: Gradient wrt output, shape (batch, features)
        Returns:
            dX: Gradient wrt input, shape (batch, features)
        """
        if self.training and self.p > 0.0:
            return dY * self.mask
        else:
            return dY

    def update(self, lr):
        # No parameters to update
        pass
