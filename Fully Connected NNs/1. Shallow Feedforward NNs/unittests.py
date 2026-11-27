"""
Solution checks of the Shallow Feedforward NNs notebook.

Each ``check_*`` function takes the objects the student built in one task,
records the values that grading compares in ``answer`` and prints them, so
the notebook itself only shows the student's own work. The answer keys
are the ones listed in ``tests/test_task.py``.
"""
import random
from pprint import pprint
from typing import Any

import numpy as np
import torch

import source
import utils as course_utils


def check_seed_all(answer: dict[str, Any]) -> None:
    """
    Record random numbers drawn right after ``seed_all``.

    With a correct ``seed_all`` these numbers are the same on every run.

    Parameters
    ----------
    answer : dict
        The notebook's answer; the ``'seed_check'`` key is set.

    Returns
    -------
    None

    Examples
    --------
    >>> src.utils.seed.seed_all(0)
    >>> check_seed_all(answer)  # prints the drawn values
    """
    seed_check = {
        'python.random': [random.random() for _ in range(4)],
        'python.randint': [random.randint(0, 9) for _ in range(4)],
        'numpy.random.rand': np.random.rand(2, 3),
        'numpy.random.randn': np.random.randn(2, 3),
        'numpy.random.randint': np.random.randint(0, 10, size=(2, 3)),
        'torch.rand': torch.rand(2, 3),
        'torch.randn': torch.randn(2, 3),
        'torch.randint': torch.randint(0, 10, (2, 3)),
        'torch.backends.cudnn.deterministic':
            torch.backends.cudnn.deterministic,
        'torch.cuda.is_available': torch.cuda.is_available(),
    }

    if seed_check['torch.cuda.is_available']:
        seed_check['torch.cuda.initial_seed'] = torch.cuda.initial_seed()
        seed_check['torch.cuda.rand'] = torch.rand(
            2, 3, device='cuda').cpu()
        seed_check['torch.cuda.randn'] = torch.randn(
            2, 3, device='cuda').cpu()
        seed_check['torch.cuda.randint'] = torch.randint(
            0, 10, (2, 3), device='cuda').cpu()

    # Resolved at call time: the teacher's notebook swaps in its own code.
    answer['seed_check'] = source.code.src.utils.detach_copy(seed_check)

    print("!!! THE VALUES BELOW SHOULD BE STABLE FOR DIFFERENT RUNS !!!")
    print()
    pprint(seed_check)


def check_dataset(answer: dict[str, Any],
                  train_dataset: torch.utils.data.Dataset,
                  valid_dataset: torch.utils.data.Dataset) -> None:
    """
    Record the dtype, shape and mean of the first sample of each dataset.

    Parameters
    ----------
    answer : dict
        The notebook's answer; the ``'X_train.*'``, ``'y_train.*'``,
        ``'X_valid.*'`` and ``'y_valid.*'`` keys are set.
    train_dataset, valid_dataset : torch.utils.data.Dataset
        The student's datasets; item 0 must be a dict with ``'image'``
        and ``'label'`` tensors.

    Returns
    -------
    None

    Examples
    --------
    >>> check_dataset(answer, MNIST_train, MNIST_valid)  # prints the samples
    """
    train_sample = train_dataset[0]
    valid_sample = valid_dataset[0]

    X_train = train_sample['image']
    X_valid = valid_sample['image']
    y_train = train_sample['label']
    y_valid = valid_sample['label']

    answer['X_train.dtype'] = str(X_train.dtype)
    answer['y_train.dtype'] = str(y_train.dtype)
    answer['X_valid.dtype'] = str(X_valid.dtype)
    answer['y_valid.dtype'] = str(y_valid.dtype)
    answer['X_train.shape'] = X_train.shape
    answer['X_valid.shape'] = X_valid.shape
    answer['y_train.shape'] = y_train.shape
    answer['y_valid.shape'] = y_valid.shape
    answer['X_train.mean'] = float(X_train.mean())
    answer['y_train.mean'] = float(y_train.float().mean())
    answer['X_valid.mean'] = float(X_valid.mean())
    answer['y_valid.mean'] = float(y_valid.float().mean())

    print('\nX_train dtype:', X_train.dtype, '\nX_valid dtype:', X_valid.dtype,
          '\ny_train dtype', y_train.dtype, '\ny_valid dtype:', y_valid.dtype)
    print('\nX_train shape:', X_train.shape, '\nX_valid shape', X_valid.shape,
          '\ny_train shape:', y_train.shape, '\ny_valid shape:', y_valid.shape)

    print()
    print('X train:', X_train)
    print('X_valid:', X_valid)
    print('y_train:', y_train)
    print('y_valid:', y_valid)


def check_network(answer: dict[str, Any], network: torch.nn.Module) -> None:
    """
    Pass a random batch of 10 images through the network and record it.

    Parameters
    ----------
    answer : dict
        The notebook's answer; the ``'check_batch'`` key is set to the
        batch dict after the forward pass (with ``'signals'`` and
        ``'postprocessed'`` filled in by the network). Task 3's check
        reuses this batch.
    network : torch.nn.Module
        The student's network, called on the batch dict.

    Returns
    -------
    None

    Examples
    --------
    >>> check_network(answer, simple_network)  # prints the output shapes
    """
    # Same keys as MNISTSimpleDataset, to see how a batch goes through the network
    check_batch = {'data': {
        'image': torch.randn(10, 28, 28),
        'label': torch.randint(low=0, high=10, size=(10,)),
    }}
    network(check_batch)

    print('output:', check_batch['signals']['output'].shape)
    print('classes:', check_batch['postprocessed']['class'].shape)

    pprint(check_batch)
    answer['check_batch'] = check_batch


def check_training_setup(answer: dict[str, Any],
                         losses: dict,
                         metrics: dict,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Any) -> None:
    """
    Record the losses, metrics, optimizer and scheduler of Task 3.

    The losses and metrics are evaluated on the batch recorded by
    ``check_network``, so that check must run first.

    Parameters
    ----------
    answer : dict
        The notebook's answer; it must hold ``'check_batch'``. The
        ``'loss_names'``, ``'metric_names'``, ``'optimizer_*'``,
        ``'scheduler_*'``, ``'test_loss_val'`` and ``'test_metrics_val'``
        keys are set.
    losses, metrics : dict
        Name -> function of a batch dict.
    optimizer : torch.optim.Optimizer
        The optimizer of the network's parameters.
    scheduler : torch.optim.lr_scheduler.LRScheduler or ReduceLROnPlateau
        The learning-rate scheduler of ``optimizer``.

    Returns
    -------
    None

    Examples
    --------
    >>> check_training_setup(answer, losses, metrics, optimizer, scheduler)
    """
    check_batch = answer['check_batch']

    answer['loss_names'] = list(losses.keys())
    answer['metric_names'] = list(metrics.keys())

    answer['optimizer_class'] = optimizer.__class__.__name__
    answer['optimizer_defaults'] = optimizer.defaults
    answer['optimizer_param_groups_count'] = len(optimizer.param_groups)
    answer['optimizer_lrs'] = [group['lr'] for group in optimizer.param_groups]

    answer['scheduler_class'] = scheduler.__class__.__name__
    answer['scheduler_optimizer_class'] = (
        scheduler.optimizer.__class__.__name__)

    answer['test_loss_val'] = {
        name: function(check_batch) for name, function in losses.items()}
    answer['test_metrics_val'] = {
        name: function(check_batch) for name, function in metrics.items()}

    print(losses)
    print(optimizer)
    print(scheduler)
    print(metrics)


def check_dataloaders(answer: dict[str, Any],
                      train_dataloader: torch.utils.data.DataLoader,
                      valid_dataloader: torch.utils.data.DataLoader) -> None:
    """
    Record the first batch of each dataloader.

    Parameters
    ----------
    answer : dict
        The notebook's answer; the ``'train_batch'`` and ``'valid_batch'``
        keys are set to the batches' text form.
    train_dataloader, valid_dataloader : torch.utils.data.DataLoader
        The student's dataloaders.

    Returns
    -------
    None

    Examples
    --------
    >>> check_dataloaders(answer, train_dl, valid_dl)  # prints both batches
    """
    train_batch = next(iter(train_dataloader))
    valid_batch = next(iter(valid_dataloader))

    pprint(train_batch)
    pprint(valid_batch)

    answer['train_batch'] = str(train_batch)
    answer['valid_batch'] = str(valid_batch)


def check_training(answer: dict[str, Any],
                   training_history: dict[str, dict[str, list]]) -> None:
    """
    Record the loss and accuracy histories and save the answer for grading.

    Parameters
    ----------
    answer : dict
        The notebook's answer; the ``'train_loss_history'``,
        ``'valid_loss_history'``, ``'train_acc_history'`` and
        ``'valid_acc_history'`` keys are set, then the whole answer is
        saved with ``utils.checker.save_answer``.
    training_history : dict
        The result of ``src.train_loop.train_model``, with the ``'ce'``
        loss and the ``'accuracy'`` metric.

    Returns
    -------
    None

    Examples
    --------
    >>> check_training(answer, training_history)
    """
    answer['train_loss_history'] = training_history['train_loss']['ce']
    answer['valid_loss_history'] = training_history['valid_loss']['ce']
    answer['train_acc_history'] = training_history['train_metrics']['accuracy']
    answer['valid_acc_history'] = training_history['valid_metrics']['accuracy']

    course_utils.checker.save_answer(answer)
