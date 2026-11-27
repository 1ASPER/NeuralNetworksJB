"""A vision transformer for MNIST."""
import torch

from .attention import TransformerBlock


class PatchEmbedding(torch.nn.Module):
    """Cut ``[batch, H, W]`` images into ``patch x patch`` squares, embed each.

    Returns ``[batch, (H // patch) * (W // patch), dim]``, patches row by
    row; each patch, flattened to ``patch * patch`` pixels, goes through
    ``Linear(patch * patch, dim)``.
    """

    def __init__(self, patch=7, dim=64):
        super().__init__()
        self.patch = patch
        self.linear = torch.nn.Linear(patch * patch, dim)

    def forward(self, images):
        batch_size, height, width = images.shape
        patches = images.reshape(batch_size, -1, self.patch * self.patch)
        ## Cut the images into patches row by row (Tensor.unfold, then reshape)
        ## YOUR CODE HERE
        # -- start --
        patches = (images.unfold(1, self.patch, self.patch).unfold(2, self.patch, self.patch)
                   .reshape(batch_size, -1, self.patch * self.patch))
        # -- end --
        return self.linear(patches)


class VisionTransformer(torch.nn.Module):
    """Patch embedding, a class token, positions, transformer blocks, a head.

    The sequence is ``[class_token, patch_1, ..., patch_n]`` plus a learned
    position embedding; after ``n_layers`` (non-causal) blocks and a final
    ``LayerNorm``, the head ``Linear(dim, n_classes)`` reads the class token.

    Follows the course batch protocol: reads ``batch['data']['image']``,
    writes ``batch['signals']['output']`` and ``batch['postprocessed']['class']``.
    """

    def __init__(self, image_size=28, patch=7, dim=64, n_layers=4, n_heads=4, n_classes=10):
        super().__init__()
        n_patches = (image_size // patch) ** 2
        self.patch_embedding = PatchEmbedding(patch, dim)
        # Placeholder layers: the network runs, wrongly, until they are defined.
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embedding = torch.nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = torch.nn.Identity()
        self.norm = torch.nn.Identity()
        self.head = torch.nn.Linear(dim, n_classes)

        ## Define the position embedding (random, std 0.02), n_layers
        ## non-causal TransformerBlocks in a torch.nn.Sequential and the
        ## final LayerNorm
        ## YOUR CODE HERE
        # -- start --
        self.position_embedding = torch.nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        self.blocks = torch.nn.Sequential(
            *[TransformerBlock(dim, n_heads, causal=False) for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(dim)
        # -- end --

    def forward(self, batch):
        images = batch['data']['image']
        patches = self.patch_embedding(images)
        class_token = self.class_token.expand(len(images), -1, -1)
        signal = torch.cat([class_token, patches], dim=1) + self.position_embedding
        signal = self.head(self.norm(self.blocks(signal))[:, 0])
        batch['signals'] = {'output': signal}
        batch['postprocessed'] = {'class': signal.argmax(dim=1)}
        return batch
