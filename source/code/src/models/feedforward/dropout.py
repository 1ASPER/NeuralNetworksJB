"""Inverted dropout."""
import torch


class Dropout(torch.nn.Module):
    """Zero every signal with probability ``p`` during training.

    In training mode, every value is kept with probability ``1 - p`` and
    divided by ``1 - p`` (so its expected value does not change), or set to
    zero. In evaluation mode (``self.training`` is ``False``) the signal
    passes unchanged. ``generator`` makes the random choice repeatable.
    """

    def __init__(self, p=0.5, generator=None):
        super().__init__()
        self.p = p
        self.generator = generator

    def forward(self, signal):
        ## YOUR CODE HERE
        return signal


class SequentialClassifier(torch.nn.Module):
    """Flatten the images and apply ``layers`` one after another.

    Follows the course batch protocol: reads ``batch['data']['image']``
    (``[batch, 28, 28]``), writes ``batch['signals']['output']`` (the
    output of the last layer) and ``batch['postprocessed']['class']``.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        signal = self.layers(batch['data']['image'].flatten(start_dim=1))
        batch['signals'] = {'output': signal}
        batch['postprocessed'] = {'class': signal.argmax(dim=1)}
        return batch
