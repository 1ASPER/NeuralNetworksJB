"""A small U-Net: one class per pixel."""
import torch
import torch.nn.functional as F

from .sequential import ResidualBottleneck


def default_block(in_channels, out_channels):
    """Residual bottleneck with batch normalization (halves the image when channels change)."""
    return ResidualBottleneck(in_channels, out_channels,
                              prenormalization=torch.nn.BatchNorm2d, residual=True)


class UNet(torch.nn.Module):
    """Encoder that halves the image, decoder that restores it, skip connections.

    With ``channels = (c0, c1, ..., cL)``:

    - ``stem``: ``Conv2d(in_channels, c0, 3, padding=1)``, full resolution;
    - ``down[i]`` (``i = 1..L``): ``block(c[i-1], c[i])``, halves the image;
    - ``up[i]`` (``i = L..1``): upsample x2, concatenate with the encoder
      signal of the same size (``c[i-1]`` channels), then
      ``Conv2d(c[i] + c[i-1], c[i-1], 1)`` and ``block(c[i-1], c[i-1])``;
    - ``head``: ``Conv2d(c0, n_classes, 1)``.

    ``forward`` reads ``batch['data']['image']`` (``[batch, H, W]``) and
    writes ``batch['signals']['output']`` (``[batch, n_classes, H, W]``) and
    ``batch['postprocessed']['mask']`` (``[batch, H, W]``).
    """

    def __init__(self, block=default_block, in_channels=1, channels=(16, 32, 64), n_classes=11):
        super().__init__()
        # Placeholder layers: the network runs, wrongly, until they are defined.
        self.stem = torch.nn.Conv2d(in_channels, channels[0], kernel_size=1)
        self.down = torch.nn.ModuleList()
        self.merge = torch.nn.ModuleList()
        self.up = torch.nn.ModuleList()
        self.head = torch.nn.Conv2d(channels[0], n_classes, kernel_size=1)

        ## Define the stem, down[i], merge[i] (the 1 x 1 convolutions after
        ## the concatenation) and up[i]; down, merge and up are ModuleLists
        ## with one entry per level, listed from the top level (largest image)
        ## YOUR CODE HERE
        # -- start --
        self.stem = torch.nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        for level in range(1, len(channels)):
            self.down.append(block(channels[level - 1], channels[level]))
            self.merge.append(torch.nn.Conv2d(channels[level] + channels[level - 1],
                                              channels[level - 1], kernel_size=1))
            self.up.append(block(channels[level - 1], channels[level - 1]))
        # -- end --

    def __forward_kernel(self, signal):
        signal = signal.unsqueeze(1)          # [batch, 1, H, W]: one channel
        signal = self.stem(signal)
        skips = []
        ## Go down through self.down, keeping every signal before it is
        ## halved, then up through merge and up in reverse order
        ## YOUR CODE HERE
        # -- start --
        for down in self.down:
            skips.append(signal)
            signal = down(signal)
        for skip, merge, up in reversed(list(zip(skips, self.merge, self.up))):
            signal = F.interpolate(signal, scale_factor=2)
            signal = up(merge(torch.cat([signal, skip], dim=1)))
        # -- end --
        return self.head(signal)

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)
        batch['signals'] = {'output': signal}
        self.postprocessing(batch)
        return batch

    def postprocessing(self, batch):
        logits = batch['signals']['output']
        mask = torch.zeros(logits.shape[0], *logits.shape[2:], dtype=torch.long, device=logits.device)
        ## The predicted class of every pixel
        ## YOUR CODE HERE
        # -- start --
        mask = logits.argmax(dim=1)
        # -- end --
        batch['postprocessed'] = {'mask': mask}
