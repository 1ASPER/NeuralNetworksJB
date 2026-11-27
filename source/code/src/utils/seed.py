import torch
import numpy as np
import random

def seed_all(seed_value):
    """
    Make every random draw of an experiment repeat from run to run.

    Training uses random numbers in many places: weight initialization,
    shuffling of the data, dropout masks. Each library (Python's ``random``,
    NumPy, PyTorch, and every GPU) keeps its own random number generator.
    After ``seed_all(seed_value)`` all of them start from the same state, so
    running the same code again gives exactly the same numbers. This makes
    results comparable between runs and makes debugging much easier.

    What to implement (replace ``pass`` with your code):

    1. Seed Python's ``random`` module with ``random.seed``.
    2. Seed NumPy with ``np.random.seed``.
    3. Seed PyTorch on the CPU with ``torch.manual_seed``.
    4. Seed each device only if it is available (skip the others):

       - NVIDIA GPUs, all of them, not only the current one:
         ``torch.cuda.manual_seed_all``, when ``torch.cuda.is_available()``;
       - Apple GPU (Metal) on a MacBook: ``torch.mps.manual_seed``, when
         ``torch.backends.mps.is_available()``;
       - TPU: ``torch_xla.core.xla_model.set_rng_state``, only when the
         optional ``torch_xla`` package can be imported. Most computers do
         not have it, so catch the ``ImportError`` and go on.

    5. Ask cuDNN (the NVIDIA GPU library) to use deterministic algorithms:
       set ``torch.backends.cudnn.deterministic = True``.

    Parameters
    ----------
    seed_value : int
        Seed for every generator. The same value gives the same random
        numbers; a different value gives different ones.

    Returns
    -------
    None
        The function only changes the state of the generators.

    Examples
    --------
    >>> seed_all(0)
    >>> first = (random.random(), np.random.rand(), torch.rand(1).item())
    >>> seed_all(0)
    >>> second = (random.random(), np.random.rand(), torch.rand(1).item())
    >>> first == second
    True
    """
    pass
    ## YOUR CODE HERE
