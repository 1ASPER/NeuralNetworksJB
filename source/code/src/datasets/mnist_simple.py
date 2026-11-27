import functools

import torch
import torchvision.datasets


class SmallMNIST:
    """Tiny offline stand-in for ``torchvision.datasets.MNIST``.

    The small run uses it so that grading needs no download: random 28 x 28
    ``uint8`` images, 512 for training and 128 for validation, with labels
    0, 1, ..., 9 repeated. ``root`` and ``download`` are accepted, like in
    ``torchvision.datasets.MNIST``, and ignored.
    """

    def __init__(self, root=None, train=True, download=False, seed=0):
        generator = torch.Generator().manual_seed(seed + (0 if train else 1000))
        size = 512 if train else 128
        self.data = torch.randint(0, 256, (size, 28, 28), generator=generator,
                                  dtype=torch.uint8)
        self.targets = torch.arange(size) % 10


class MNISTSimpleDataset:
    """
    MNIST digits as a PyTorch-style dataset of ``{'image', 'label'}`` dicts.

    A PyTorch dataset needs only two methods: ``len(dataset)`` (the number
    of samples) and ``dataset[index]`` (one sample). A ``DataLoader`` then
    calls them to build shuffled mini-batches. You implement the three
    methods below; the docstring of each says exactly what it must do.

    Parameters
    ----------
    train : bool
        ``True`` for the training split (60000 images), ``False`` for the
        validation split (10000 images).
    small : bool
        ``True`` replaces MNIST by ``SmallMNIST``, a tiny random stand-in
        (512 training / 128 validation images) that needs no download.
        Grading always uses it.
    seed : int
        Seed of ``SmallMNIST``'s random images; not used for real MNIST.

    Attributes
    ----------
    X : torch.Tensor
        All images, shape ``(n_samples, 28, 28)``, dtype ``uint8``, values
        0..255 (set in ``__init__``).
    y : torch.Tensor
        All labels, shape ``(n_samples,)``, values 0..9 (set in
        ``__init__``).

    Examples
    --------
    >>> dataset = MNISTSimpleDataset(train=True, small=True)
    >>> len(dataset)
    512
    >>> sample = dataset[0]
    >>> sample['image'].shape, sample['image'].dtype, sample['label'].dtype
    (torch.Size([28, 28]), torch.float32, torch.int64)
    """

    def __init__(self, train=True, small=False, seed=0):
        """
        Load the images and labels of one split into ``self.X`` and ``self.y``.

        What to implement (replace the zero placeholders):

        1. Create the dataset object:
           ``mnist_class(root='~/', train=train, download=True)``.
           ``mnist_class`` is already chosen for you: real MNIST, or
           ``SmallMNIST`` in the small run. ``root='~/'`` keeps the download
           out of the course folder; ``train`` picks the split.
        2. Store its images (the ``.data`` field) in ``self.X`` and its
           labels (the ``.targets`` field) in ``self.y``.

        Parameters
        ----------
        train : bool
            Which split to load: training (``True``) or validation.
        small : bool
            Use the offline ``SmallMNIST`` instead of real MNIST.
        seed : int
            Seed of ``SmallMNIST``.

        Returns
        -------
        None
        """
        # A small run reads the tiny offline stand-in instead of real MNIST.
        if small:
            mnist_class = functools.partial(SmallMNIST, seed=seed)
        else:
            mnist_class = torchvision.datasets.MNIST

        self.X = torch.zeros((32, 28, 28), dtype=torch.uint8)
        self.y = torch.zeros(32, dtype=torch.long)
        ## Load MNIST with mnist_class here
        ## YOUR CODE HERE


    def __len__(self):
        """
        Return the number of samples, so that ``len(dataset)`` works.

        What to implement: return the number of labels (or images) loaded
        in ``__init__``, instead of the placeholder 32.

        Returns
        -------
        int
            Number of samples in this split (512 in the small training run).
        """
        res = 32
        ## Return number of items that is there in the dataset
        ## YOUR CODE HERE
        return res


    def __getitem__(self, index):
        """
        Return sample number ``index``, so that ``dataset[index]`` works.

        What to implement (replace the placeholder values in ``sample``):

        - ``sample['image']``: image ``index`` of ``self.X`` as ``float32``,
          scaled from ``[0, 255]`` to ``[-1, 1]``. Values centred at zero
          make the optimization easier than raw pixel values.
        - ``sample['label']``: label ``index`` of ``self.y`` as ``long``
          (``int64``), the type ``cross_entropy`` expects for class indices.

        Parameters
        ----------
        index : int
            Position of the sample, ``0 <= index < len(self)``.

        Returns
        -------
        dict
            ``{'image': tensor of shape (28, 28), float32, values in
            [-1, 1]; 'label': scalar tensor, int64, value 0..9}``.
        """
        sample = {
            'image': self.X[index].float(),
            'label': self.y[index].long(),
        }

        ## Return a sample of the dataset that corresponds to the input index
        ## YOUR CODE HERE
        
        return sample 
