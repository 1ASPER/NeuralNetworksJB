"""MNIST digits scattered on larger canvases, for segmentation and detection."""
import numpy as np
import torch
import torchvision.datasets

from .mnist_simple import SmallMNIST

MAX_DIGITS = 4
DIGIT_THRESHOLD = 0.3


def make_canvases(images, labels, n_canvases, size=64, max_digits=MAX_DIGITS, seed=0):
    """Place 1..``max_digits`` MNIST digits on each canvas, boxes never overlap.

    Parameters
    ----------
    images : torch.Tensor
        ``uint8`` digits, shape ``[N, 28, 28]``.
    labels : torch.Tensor
        Digit classes, shape ``[N]``.
    n_canvases : int
        Number of canvases to make.
    size : int
        Canvas height and width in pixels.
    max_digits : int
        Largest number of digits on one canvas.
    seed : int
        Seed of the random placement.

    Returns
    -------
    dict
        ``image``: ``float32 [n, size, size]`` in ``[0, 1]``;
        ``mask``: ``int64 [n, size, size]``, 0 for background and
        ``digit + 1`` for the pixels of a digit (brighter than 0.3);
        ``boxes``: ``float32 [n, max_digits, 4]``, ``(x1, y1, x2, y2)``
        tight around the digit pixels, rows of zeros for missing digits;
        ``labels``: ``int64 [n, max_digits]``, ``-1`` for missing digits.
    """
    generator = np.random.default_rng(seed)
    digits = images.numpy().astype(np.float32) / 255.0
    classes = labels.numpy()
    canvas = np.zeros((n_canvases, size, size), np.float32)
    mask = np.zeros((n_canvases, size, size), np.int64)
    boxes = np.zeros((n_canvases, max_digits, 4), np.float32)
    box_labels = np.full((n_canvases, max_digits), -1, np.int64)
    for canvas_index in range(n_canvases):
        placed = 0
        for _ in range(generator.integers(1, max_digits + 1)):
            for _attempt in range(20):
                digit_index = generator.integers(len(digits))
                digit = digits[digit_index]
                rows, columns = np.nonzero(digit > DIGIT_THRESHOLD)
                if len(rows) == 0:
                    break
                top, left = generator.integers(0, size - 28 + 1, size=2)
                box = [left + columns.min(), top + rows.min(),
                       left + columns.max() + 1, top + rows.max() + 1]
                if any(_overlap(box, other) for other in boxes[canvas_index, :placed]):
                    continue
                window = (slice(top, top + 28), slice(left, left + 28))
                canvas[canvas_index][window] = np.maximum(canvas[canvas_index][window], digit)
                mask[canvas_index][window][digit > DIGIT_THRESHOLD] = classes[digit_index] + 1
                boxes[canvas_index, placed] = box
                box_labels[canvas_index, placed] = classes[digit_index]
                placed += 1
                break
    return {
        'image': torch.from_numpy(canvas),
        'mask': torch.from_numpy(mask),
        'boxes': torch.from_numpy(boxes),
        'labels': torch.from_numpy(box_labels),
    }


def _overlap(box, other):
    return not (box[2] <= other[0] or other[2] <= box[0]
                or box[3] <= other[1] or other[3] <= box[1])


def load_canvases(train=True, small=False, seed=0, n_canvases=None):
    """Canvases made from MNIST, or from its tiny offline stand-in.

    ``small=True`` uses ``SmallMNIST`` (random digits, no download) and 256
    training or 64 validation canvases; otherwise real MNIST (downloaded to
    ``~/`` once) and 10 000 or 1 000 canvases. ``n_canvases`` overrides the
    number. Returns the dictionary of ``make_canvases``.
    """
    if small:
        mnist = SmallMNIST(train=train, seed=seed)
    else:
        mnist = torchvision.datasets.MNIST(root='~/', train=train, download=True)
    if n_canvases is None:
        n_canvases = (256 if train else 64) if small else (10000 if train else 1000)
    return make_canvases(mnist.data, mnist.targets, n_canvases,
                         seed=seed + (0 if train else 1))


class MNISTCanvasDataset:
    """Canvases for segmentation: an image and its mask of digit classes.

    ``canvases`` is the dictionary returned by ``load_canvases``. An item is
    ``{'image': float [64, 64] in [-1, 1], 'mask': int64 [64, 64]}``; the
    image is scaled like in ``MNISTSimpleDataset``: ``x * 2 - 1``.
    """

    def __init__(self, canvases):
        self.canvases = canvases

    def __len__(self):
        res = 1
        ## Return the number of canvases
        ## YOUR CODE HERE
        return res

    def __getitem__(self, index):
        sample = {
            'image': torch.zeros(64, 64),
            'mask': torch.zeros(64, 64, dtype=torch.long),
        }
        ## Return the canvas number index: the image scaled to [-1, 1]
        ## and its mask
        ## YOUR CODE HERE
        return sample


def cell_centers(size=64, stride=2):
    """Centres ``(x, y)`` of the output cells, row by row: ``[(size // stride) ** 2, 2]``."""
    n_cells = size // stride
    rows, columns = torch.meshgrid(torch.arange(n_cells), torch.arange(n_cells), indexing='ij')
    centers = torch.stack([columns, rows], dim=-1).reshape(-1, 2).float()
    return centers * stride + stride / 2


def assign_targets(boxes, labels, centers, n_classes=10, center_fraction=0.25):
    """Targets of every output cell for one image (anchor-free RetinaNet).

    A cell is **positive** for a box when its centre lies in the central part
    of the box: ``|center - box_center| <= max(center_fraction * box_size,
    1)`` along both axes. If a cell is positive for several boxes, the
    smallest box wins.

    Parameters
    ----------
    boxes : torch.Tensor
        ``[K, 4]`` boxes ``(x1, y1, x2, y2)``.
    labels : torch.Tensor
        ``[K]`` classes; ``-1`` marks a missing box, to be ignored.
    centers : torch.Tensor
        ``[C, 2]`` cell centres from ``cell_centers``.
    n_classes : int
        Number of classes.
    center_fraction : float
        Half-size of the central part, as a fraction of the box size.

    Returns
    -------
    tuple of torch.Tensor
        ``class_target``: ``float [C, n_classes]``, one-hot for positive
        cells, all zeros otherwise; ``box_target``: ``float [C, 4]``, the
        box of positive cells, zeros otherwise.
    """
    class_target = torch.zeros(len(centers), n_classes)
    box_target = torch.zeros(len(centers), 4)
    ## YOUR CODE HERE
    return class_target, box_target


class MNISTDetectionDataset(MNISTCanvasDataset):
    """Canvases for detection: an image, its true boxes and the cell targets.

    An item is ``{'image', 'boxes' [4, 4], 'labels' [4], 'class_target'
    [1024, 10], 'box_target' [1024, 4]}``: the targets come from
    ``assign_targets`` for the 32 x 32 cells of stride 2. Missing boxes have
    label ``-1``.
    """

    def __init__(self, canvases, stride=2):
        super().__init__(canvases)
        self.centers = cell_centers(size=canvases['image'].shape[-1], stride=stride)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        boxes = self.canvases['boxes'][index]
        labels = self.canvases['labels'][index]
        class_target, box_target = assign_targets(boxes, labels, self.centers)
        return {
            'image': sample['image'],
            'boxes': boxes,
            'labels': labels,
            'class_target': class_target,
            'box_target': box_target,
        }
