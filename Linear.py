import torch
class Linear:
    """
    A simple fully connected (dense) layer.
    Performs a linear transformation:  Z = XW + b
    """

    def __init__(self, nin, nout, device="cpu"):
        """
        Initialize the layer parameters.
        """
        # Initialize weights from a normal distribution
        self.W = torch.randn(nin, nout, device=device, requires_grad=False)
        # Initialize biases to zero
        self.b = torch.zeros(nout, device=device, requires_grad=False)
        self.training = True  # for compatibility with Dropout/BatchNorm

    def train(self):
        """Switch to training mode."""
        self.training = True
        return self

    def eval(self):
        """Switch to evaluation mode."""
        self.training = False
        return self

    def forward(self, X):
        """
        Forward pass: compute the output of the layer.
        """
        self.X = X  # store for backward pass
        # TODO: Implement Z = XW + b
        Z = torch.matmul(self.X,self.W) + self.b
        return Z
    def backward(self, dZ):
        """
        Backward pass: compute gradients w.r.t. W, b, and X.
        """
        # Gradiente respecto a X (para propagar hacia capas anteriores)
        self.dX = torch.matmul(dZ,self.W.T)
        # Gradiente respecto a W
        self.dW = torch.matmul(self.X.T,dZ)
        # Gradiente respecto a b (sumar sobre batch)
        self.db = dZ.sum(axis=0, keepdims=True)

        return self.dX

    def update(self, lr):
        """
        Update parameters using gradient descent.
        """
        self.W -= lr * self.dW
        #print(self.db)
        self.b = self.b - lr * self.db
