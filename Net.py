class Net:
    """
    A simple sequential container for custom layers.
    Provides PyTorch-like train()/eval() switches and
    runs forward/backward/update across all layers.
    """

    def __init__(self):
        """
        Start with an empty list of layers and set the network
        to training mode by default.
        """
        self.layers = []
        self.training = True  # True = training mode, False = eval mode

    def add(self, layer):
        """
        Add a layer to the network.

        Args:
            layer: Any object that implements forward(), backward(), update(),
                   and (optionally) train()/eval() for mode control.
        """
        self.layers.append(layer)

    # ---- Mode control (pro-style) ----
    def train(self):
        """
        Switch the whole network to training mode and propagate
        the setting to layers that implement train().
        """
        self.training = True
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()
        return self

    def eval(self):
        """
        Switch the whole network to evaluation mode and propagate
        the setting to layers that implement eval().
        """
        self.training = False
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()
        return self

    # ---- Core passes ----
    def forward(self, X):
        """
        Forward pass through all layers.

        Args:
            X (torch.Tensor): Input to the network.

        Returns:
            torch.Tensor: Output after the last layer.
        """
        for layer in self.layers:
            #Implement the forward pass
            X = layer.forward(X)  # output of one layer becomes input to the next
        return X

    def backward(self, dZ):
        """
        Backward pass through all layers in reverse order.

        Args:
            dZ (torch.Tensor): Gradient of the loss w.r.t. network output.

        Returns:
            torch.Tensor: Gradient of the loss w.r.t. the network input.
        """
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)  # cada capa calcula gradientes y devuelve dX
        return dZ

    def update(self, lr):
        """
        Update parameters of all trainable layers with the given learning rate.

        Args:
            lr (float): Learning rate.
        """
        for layer in self.layers:
            # Some layers (e.g., activations) may not have parameters
            if hasattr(layer, "update"):
                layer.update(lr)