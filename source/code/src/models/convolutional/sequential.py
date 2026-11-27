"""A fully convolutional network with residual bottleneck blocks."""
import torch


class ResidualBottleneck(torch.nn.Module):
    """Two 3 x 3 convolutions, optionally with a residual connection.

    ``block`` = prenormalization -> Conv 3x3 (in -> out // compression) ->
    postnormalization -> activation -> prenormalization -> Conv 3x3
    (-> out) -> postnormalization. When ``in_channels != out_channels``, the
    block also halves the image (stride 2 in its second convolution) and
    the residual connection goes through ``bypass``, a 1 x 1 convolution
    with stride 2; otherwise ``bypass`` is the identity.

    ``prenormalization`` and ``postnormalization`` take a number of
    channels and return a module, e.g. ``torch.nn.BatchNorm2d``.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            prenormalization=lambda n_channels: torch.nn.Identity(),
            postnormalization=lambda n_channels: torch.nn.Identity(),
            activation=torch.nn.ReLU,
            compression=1,
            residual=True):
        super().__init__()
        self.residual = residual
        stride = 2 if out_channels != in_channels else 1
        # Placeholder layers: the block runs, wrongly, until they are defined.
        self.block = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bypass = torch.nn.Identity()

        ## Define self.block and, when the channels change, self.bypass
        ## YOUR CODE HERE

    def forward(self, signal):
        output = self.block(signal)
        ## With self.residual, return block(signal) + bypass(signal)
        ## YOUR CODE HERE
        return output


class FullyConvolutionalNN(torch.nn.Module):
    """Encoder, levels of blocks, global average pooling, linear classifier.

    Level ``i`` has ``n_blocks[i]`` blocks with ``mid_channels[i]`` channels;
    between two levels, one more block changes the channels to the next
    level's (and halves the image). ``block(in_channels, out_channels)``
    makes one block.

    ``forward`` follows the course batch protocol: it reads
    ``batch['data']['image']`` (``[batch, 28, 28]``) and writes
    ``batch['signals']['output']`` and ``batch['postprocessed']['class']``.
    """

    def __init__(
            self,
            block=lambda in_channels, out_channels: torch.nn.Conv2d(in_channels, out_channels, (1, 1)),
            in_channels=1,
            mid_channels=(16, 32, 64, 128),
            out_channels=10,
            n_blocks=(1, 1, 1, 1)):
        super().__init__()
        # Placeholder layers: the network runs, wrongly, until they are defined.
        self.encoder = torch.nn.Conv2d(in_channels, mid_channels[-1], kernel_size=1)
        self.blocks = torch.nn.Identity()
        self.decoder = torch.nn.Linear(mid_channels[-1], out_channels)

        ## Define the encoder (Conv 3x3, in_channels -> mid_channels[0]),
        ## the blocks of all levels with a channel-changing block between
        ## two levels, and the decoder (Linear, mid_channels[-1] -> out_channels)
        ## YOUR CODE HERE

    def __forward_kernel(self, signal):
        signal = signal.unsqueeze(1)          # [batch, 1, 28, 28]: one channel
        signal = self.encoder(signal)
        signal = self.blocks(signal)
        signal = signal.mean(dim=(-2, -1))    # global average pooling
        signal = self.decoder(signal)
        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)

        # Put the result into the batch
        batch['signals'] = {'output': signal}

        # Perform postprocessing after we get the output
        self.postprocessing(batch)

        return batch

    def postprocessing(self, batch):
        logits = batch['signals']['output']
        signal = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        ## Turn the logits into predicted classes
        ## YOUR CODE HERE
        batch['postprocessed'] = {'class': signal}
