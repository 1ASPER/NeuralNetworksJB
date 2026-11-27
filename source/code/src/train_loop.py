from html import escape

import torch
from tqdm.auto import tqdm

from .utils import mlflow as mlflow_utils


def _value_text(value):
    return '-' if value is None else f'{value:8.4f}'


def _number(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'item'):
        return value.item()
    return value


def _model_device(model):
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device

    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device

    return torch.device('cpu')


def _move_to_device(value, device):
    if torch.is_tensor(value) and value.dtype == torch.float64 and device.type == 'mps':
        # The Apple GPU has no float64.
        return value.to(device=device, dtype=torch.float32)
    if hasattr(value, 'to'):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _normalize_named_functions(functions, default_name):
    if isinstance(functions, dict):
        return functions
    return {default_name: functions}


def _loss_metric_name(stage, loss_name):
    if loss_name == 'loss':
        return f'{stage}_loss'
    return f'{stage}_loss/{loss_name}'


def _batch_status(batch_losses, batch_loss_emas):
    lines = ['batch losses:']
    if not batch_losses:
        lines.append('-')
    else:
        for loss_name, loss_value in batch_losses.items():
            loss_ema = None if batch_loss_emas is None else batch_loss_emas.get(loss_name)
            lines.append(
                f'{loss_name}:\t train {_value_text(loss_value)} | ema {_value_text(loss_ema)}'
            )
    return '\n'.join(lines)


def _set_bar_message(bar, message):
    children = getattr(getattr(bar, 'container', None), 'children', ())
    if len(children) >= 3:
        container = bar.container
        _, progress_widget, right_text = children

        container.layout.flex_flow = 'row wrap'
        container.layout.align_items = 'center'
        progress_widget.layout.flex = '1 1 auto'
        right_text.layout.width = '100%'
        right_text.layout.flex = '0 0 100%'
        right_text.layout.margin = '2px 0 0 0'

        bar.set_postfix_str('', refresh=True)
        progress_text = bar.format_meter(
            bar.n,
            bar.total,
            bar.format_dict.get('elapsed', 0),
        )
        right_text.value = (
            f'<div>{progress_text}</div>'
            '<pre style="margin:2px 0 0 0;font-family:monospace;white-space:pre;">'
            f'{escape(message)}'
            '</pre>'
        )
    else:
        bar.set_postfix_str(message.replace('\n', ' | '))


def _keep_bar_message_on_close(bar, get_message):
    original_close = bar.close

    def close_with_message():
        original_close()
        message = get_message()
        if message is not None:
            _set_bar_message(bar, message)

    bar.close = close_with_message


def _epoch_status(valid_losses, train_losses, valid_metrics=None, train_metrics=None):
    lines = [f'epoch loss:']

    loss_names = list((valid_losses or {}).keys())
    loss_names += [
        loss_name
        for loss_name in (train_losses or {})
        if loss_name not in loss_names
    ]
    if not loss_names:
        lines.append('-')
    else:
        for loss_name in loss_names:
            valid_loss = None if valid_losses is None else valid_losses.get(loss_name)
            train_loss = None if train_losses is None else train_losses.get(loss_name)
            lines.append(
                f'{loss_name}:\t val {_value_text(valid_loss)} | train {_value_text(train_loss)}'
            )

    lines.append(f'epoch metrics:')

    if not valid_metrics and not train_metrics:
        lines.append('-')
    else:
        metric_names = list((valid_metrics or {}).keys())
        metric_names += [
            metric_name
            for metric_name in (train_metrics or {})
            if metric_name not in metric_names
        ]
        for metric_name in metric_names:
            valid_metric = None if valid_metrics is None else valid_metrics.get(metric_name)
            train_metric = None if train_metrics is None else train_metrics.get(metric_name)
            lines.append(
                f'{metric_name}:\t val {_value_text(valid_metric)} | train {_value_text(train_metric)}'
            )

    return '\n'.join(lines)


def _empty_metric_totals(metrics):
    return {
        metric_name: {'enumerator': 0.0, 'denominator': 1.0e-8}
        for metric_name in metrics
    }


def _empty_loss_totals(losses):
    return {
        loss_name: {'enumerator': 0.0, 'denominator': 1.0e-8}
        for loss_name in losses
    }


def _compute_losses(losses, batch):
    signal = next(iter(batch['signals'].values()))
    results = {
        loss_name: signal.sum() * 0
        for loss_name in losses
    }
    ## YOUR CODE HERE
    return results


def _sum_losses(batch_losses):
    total_loss = None
    for loss_value in batch_losses.values():
        pass
        ## YOUR CODE HERE
    if total_loss is None:
        return next(iter(batch_losses.values())).sum() * 0
    return total_loss


def _update_loss_totals(loss_totals, batch_losses):
    for loss_name, loss_value in batch_losses.items():
        loss_totals[loss_name]['enumerator'] += _number(loss_value)
        loss_totals[loss_name]['denominator'] += 1.0


def _update_loss_emas(loss_emas, batch_losses, beta=0.90):
    for loss_name, loss_value in batch_losses.items():
        loss_number = _number(loss_value)
        if loss_emas[loss_name] is None:
            loss_emas[loss_name] = loss_number
        else:
            loss_emas[loss_name] = beta * loss_emas[loss_name] + (1 - beta) * loss_number


def _finalize_loss_totals(loss_totals):
    return {
        loss_name: totals['enumerator'] / totals['denominator']
        for loss_name, totals in loss_totals.items()
    }


def _update_metric_totals(metric_totals, metrics, batch):
    
    
    for metric_name, metric_fn in metrics.items():
        pass
        ## YOUR CODE HERE


def _finalize_metric_totals(metric_totals):
    return {
        metric_name: _number(totals['enumerator'] / totals['denominator'])
        for metric_name, totals in metric_totals.items()
    }


def _move_model_to_device(model, device):
    """Move ``model`` to ``device``; return ``(original_device, device)``."""
    original_device = _model_device(model)
    if device is None:
        return original_device, original_device
    device = torch.device(device)
    model.to(device)
    return original_device, device


def _check_losses(loss):
    """Return ``loss`` as a non-empty dict of name -> loss function."""
    losses = _normalize_named_functions(loss, 'loss')
    if not isinstance(losses, dict):
        raise TypeError('loss must be a function or a dictionary mapping loss names to loss functions')
    if not losses:
        raise ValueError('loss must contain at least one loss function')
    return losses


def _check_metrics(metrics):
    """Return ``metrics`` as a dict of name -> metric function."""
    metrics = {} if metrics is None else metrics
    if not isinstance(metrics, dict):
        raise TypeError('metrics must be a dictionary mapping metric names to metric functions')
    return metrics


def _build_history(losses, metrics):
    """Return the empty per-epoch history of every loss and metric."""
    return {
        'train_loss': {loss_name: [] for loss_name in losses},
        'valid_loss': {loss_name: [] for loss_name in losses},
        'train_metrics': {metric_name: [] for metric_name in metrics},
        'valid_metrics': {metric_name: [] for metric_name in metrics},
    }


def _prepare_batch(batch, device):
    """Wrap a dataloader batch into the batch dict and move it to ``device``."""
    return _move_to_device({'data': batch}, device)


def _finalize_epoch(train_losses, valid_losses, train_metrics, valid_metrics,
                    has_validation):
    """Turn the epoch totals into mean values, keyed like the history."""
    epoch_results = {
        'train_loss': _finalize_loss_totals(train_losses),
        'valid_loss': _finalize_loss_totals(valid_losses),
        'train_metrics': _finalize_metric_totals(train_metrics),
        'valid_metrics': _finalize_metric_totals(valid_metrics),
    }
    if not has_validation:
        epoch_results['valid_loss'] = dict.fromkeys(valid_losses)
        epoch_results['valid_metrics'] = dict.fromkeys(valid_metrics)
    return epoch_results


def _append_to_history(history, epoch_results):
    for stage, values in epoch_results.items():
        for name, value in values.items():
            history[stage][name].append(value)


def _step_scheduler(scheduler, valid_losses):
    """Step ``scheduler`` once per epoch; plateau schedulers get the validation loss."""
    if scheduler is None:
        return
    if (isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            or getattr(scheduler, 'steps_on_validation_loss', False)):
        scheduler.step(sum(valid_losses.values()))
    else:
        scheduler.step()


def _log_training_params(mlflow_logger, run_name, model, loss, optimizer,
                         scheduler, n_epochs, train_dl, valid_dl):
    if mlflow_logger is None:
        return
    mlflow_logger.log_params(mlflow_utils.collect_training_params(
        run_name,
        model,
        loss,
        optimizer,
        scheduler,
        getattr(train_dl, 'batch_size', None),
        n_epochs,
        train_dl,
        valid_dl,
    ))


def _log_batch_losses(mlflow_logger, loss_values, step):
    if mlflow_logger is None:
        return
    for loss_name, loss_value in loss_values.items():
        mlflow_logger.log_metric(
            _loss_metric_name('batch/train', loss_name),
            loss_value,
            step=step,
        )


def _log_epoch_results(mlflow_logger, epoch_results, epoch, has_validation):
    if mlflow_logger is None:
        return
    for loss_name in epoch_results['train_loss']:
        mlflow_logger.log_metric(
            _loss_metric_name('epoch/train', loss_name),
            epoch_results['train_loss'][loss_name],
            step=epoch,
        )
        if has_validation:
            mlflow_logger.log_metric(
                _loss_metric_name('epoch/valid', loss_name),
                epoch_results['valid_loss'][loss_name],
                step=epoch,
            )

    for metric_name in epoch_results['train_metrics']:
        mlflow_logger.log_metric(
            f'epoch/train_{metric_name}',
            epoch_results['train_metrics'][metric_name],
            step=epoch,
        )
        if has_validation:
            mlflow_logger.log_metric(
                f'epoch/valid_{metric_name}',
                epoch_results['valid_metrics'][metric_name],
                step=epoch,
            )


def _plot_history(mlflow_logger, history, run_name):
    if mlflow_logger is not None:
        mlflow_utils.plot_training_curves(mlflow_logger, history, run_name)


class _ProgressBars:
    """Epoch and batch progress bars, each with a loss summary under it.

    The summaries stay visible after the bars close.
    """

    def __init__(self, n_epochs, n_batches):
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.epoch_message = None
        self.batch_message = None

    def __enter__(self):
        self.epoch_bar = tqdm(total=self.n_epochs, desc='epochs', position=0)
        _keep_bar_message_on_close(self.epoch_bar, lambda: self.epoch_message)
        self.batch_bar = tqdm(total=self.n_batches, desc='batches', position=1, leave=True)
        _keep_bar_message_on_close(self.batch_bar, lambda: self.batch_message)
        return self

    def __exit__(self, *exception_info):
        self.batch_bar.close()
        self.epoch_bar.close()
        if self.epoch_message is not None:
            _set_bar_message(self.epoch_bar, self.epoch_message)
        if self.batch_message is not None:
            _set_bar_message(self.batch_bar, self.batch_message)
        return False

    def start_epoch(self, epoch):
        epoch_width = len(str(self.n_epochs))
        self.batch_bar.reset(total=self.n_batches)
        self.batch_bar.set_description_str(f'epoch {epoch + 1:>{epoch_width}}/{self.n_epochs}')
        self.batch_message = _batch_status(None, None)
        _set_bar_message(self.batch_bar, self.batch_message)

    def show_batch(self, loss_values, loss_emas):
        self.batch_bar.update(1)
        self.batch_message = _batch_status(loss_values, loss_emas)
        _set_bar_message(self.batch_bar, self.batch_message)

    def show_epoch(self, epoch_results):
        self.epoch_bar.update(1)
        self.epoch_message = _epoch_status(
            epoch_results['valid_loss'],
            epoch_results['train_loss'],
            epoch_results['valid_metrics'],
            epoch_results['train_metrics'],
        )
        _set_bar_message(self.epoch_bar, self.epoch_message)


def train_model(
    model,
    n_epochs,
    train_dl,
    valid_dl,
    loss,
    optimizer,
    metrics=None,
    scheduler=None,
    mlflow_logger=None,
    run_name='model',
    device=None,
):
    # The model trains on `device` and returns to where it was, so that the
    # code after the training keeps working with tensors on that device.
    original_device, device = _move_model_to_device(model, device)
    losses = _check_losses(loss)
    metrics = _check_metrics(metrics)
    history = _build_history(losses, metrics)
    has_validation = valid_dl is not None
    _log_training_params(mlflow_logger, run_name, model, loss, optimizer,
                         scheduler, n_epochs, train_dl, valid_dl)

    with _ProgressBars(n_epochs, len(train_dl)) as progress:
        for epoch in range(n_epochs):
            train_losses = _empty_loss_totals(losses)
            valid_losses = _empty_loss_totals(losses)
            loss_emas = {loss_name: None for loss_name in losses}
            train_metrics = _empty_metric_totals(metrics)
            valid_metrics = _empty_metric_totals(metrics)
            progress.start_epoch(epoch)

            for batch_index, batch in enumerate(train_dl):
                batch = _prepare_batch(batch, device)

                ## YOUR CODE HERE
                # Implement one training step:
                # switch to training mode
                # reset gradients
                # run the model
                # compute named losses
                # store named losses and their sum
                # backpropagate through the summed loss
                # update weights,
                # switch the model to evaluation mode
                # update metric numerators/denominators.
                loss_values = {
                    loss_name: torch.tensor(0.0)
                    for loss_name in losses
                }

                _update_loss_totals(train_losses, loss_values)
                _update_loss_emas(loss_emas, loss_values)
                progress.show_batch(loss_values, loss_emas)
                _log_batch_losses(mlflow_logger, loss_values,
                                  step=epoch * len(train_dl) + batch_index)

            with torch.no_grad():
                # valid_dl=None trains without a validation set.
                for valid_batch in (valid_dl if has_validation else []):
                    valid_batch = _prepare_batch(valid_batch, device)

                    ## YOUR CODE HERE
                    # Implement one validation step:
                    # switch the model to evaluation mode
                    # run the model without gradients
                    # compute and store named validation losses
                    # and update metric numerators/denominators. 
                    # Do not call backward() or step().
                    valid_loss_values = {
                        loss_name: torch.tensor(0.0)
                        for loss_name in losses
                    }

                    _update_loss_totals(valid_losses, valid_loss_values)

            epoch_results = _finalize_epoch(train_losses, valid_losses,
                                            train_metrics, valid_metrics,
                                            has_validation)
            _append_to_history(history, epoch_results)
            _log_epoch_results(mlflow_logger, epoch_results, epoch, has_validation)
            _step_scheduler(scheduler, epoch_results['valid_loss'])
            progress.show_epoch(epoch_results)

    model.to(original_device)
    _plot_history(mlflow_logger, history, run_name)
    return history
