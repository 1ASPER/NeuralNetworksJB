"""Anchor-free RetinaNet: one prediction per output cell of 2 x 2 pixels."""
import math

import torch
import torch.nn.functional as F
import torchvision

from ...datasets.mnist_canvas import cell_centers
from .segmentation import default_block


def head(channels, n_outputs):
    """Two 3 x 3 convolutions: the classification or the box head of RetinaNet."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(channels, n_outputs, kernel_size=3, padding=1),
    )


class AnchorFreeRetinaNet(torch.nn.Module):
    """RetinaNet without anchors on a feature pyramid of stride 2.

    - ``stem``: ``Conv2d(in_channels, c0, 3, stride=2, padding=1)``, then
      ``levels[0] = block(c0, c0)``; ``levels[i] = block(c[i-1], c[i])``
      halves the image again (strides 2, 4, 8, 16 for four channels; the
      channels must grow, a block halves the image only when they change);
    - ``lateral[i]``: ``Conv2d(c[i], width, 1)``; the pyramid goes from the
      smallest level up: ``p = lateral[i](x_i) + upsample(p)``, ending at
      stride 2;
    - ``class_head`` and ``box_head``: ``head(width, n_classes)`` and
      ``head(width, 4)``.

    Every output cell predicts ``n_classes`` logits and four distances
    ``(left, top, right, bottom)`` from its centre to the sides of its box.

    ``forward`` reads ``batch['data']['image']`` (``[batch, H, W]``) and
    writes ``batch['signals']``: ``class_logits [batch, cells, n_classes]``,
    ``boxes [batch, cells, 4]``; then ``postprocessing`` writes
    ``batch['postprocessed']``: ``boxes [batch, max_detections, 4]``,
    ``scores`` and ``labels`` ``[batch, max_detections]`` (label ``-1`` and
    score 0 for empty rows).
    """

    def __init__(self, block=default_block, in_channels=1, channels=(16, 32, 64, 128), width=64,
                 n_classes=10, box_scale=8.0, prior=0.01, score_threshold=0.3,
                 iou_threshold=0.5, max_detections=10):
        super().__init__()
        self.n_classes = n_classes
        self.box_scale = box_scale
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        self.stem = torch.nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=2, padding=1)
        self.levels = torch.nn.ModuleList(
            [block(channels[0], channels[0])]
            + [block(channels[i - 1], channels[i]) for i in range(1, len(channels))])
        # Placeholder layers: the network runs, wrongly, until they are defined.
        self.lateral = torch.nn.ModuleList([torch.nn.Conv2d(channels[0], width, kernel_size=1)])
        self.class_head = torch.nn.Conv2d(width, n_classes, kernel_size=1)
        self.box_head = torch.nn.Conv2d(width, 4, kernel_size=1)

        ## Define lateral (one 1 x 1 convolution per level), class_head and
        ## box_head; set the last bias of class_head to -log((1 - prior) / prior)
        ## YOUR CODE HERE

    def __forward_kernel(self, signal):
        signal = self.stem(signal.unsqueeze(1))
        features = []
        for level in self.levels:
            signal = level(signal)
            features.append(signal)
        pyramid = self.lateral[0](features[0])
        ## Build the pyramid from the smallest level up: lateral[i] of
        ## features[i] plus the upsampled pyramid of the level below
        ## YOUR CODE HERE
        # [batch, outputs, rows, columns] -> [batch, cells, outputs]
        class_logits = self.class_head(pyramid).flatten(2).transpose(1, 2)
        distances = F.softplus(self.box_head(pyramid)).flatten(2).transpose(1, 2) * self.box_scale
        return class_logits, distances

    def forward(self, batch):
        image = batch['data']['image']
        class_logits, distances = self.__forward_kernel(image)
        centers = cell_centers(size=image.shape[-1], stride=2).to(image.device)
        boxes = torch.zeros_like(distances)
        ## Turn the distances (left, top, right, bottom) from the cell
        ## centers into boxes (x1, y1, x2, y2)
        ## YOUR CODE HERE
        batch['signals'] = {'class_logits': class_logits, 'boxes': boxes}
        self.postprocessing(batch)
        return batch

    def postprocessing(self, batch):
        class_logits = batch['signals']['class_logits']
        boxes = batch['signals']['boxes']
        n_images = len(boxes)
        result = {
            'boxes': torch.zeros(n_images, self.max_detections, 4, device=boxes.device),
            'scores': torch.zeros(n_images, self.max_detections, device=boxes.device),
            'labels': torch.full((n_images, self.max_detections), -1, dtype=torch.long,
                                 device=boxes.device),
        }
        for image_index in range(n_images):
            pass
            ## For every cell keep the best class and its probability
            ## (sigmoid); drop cells below score_threshold; remove overlaps
            ## with torchvision.ops.batched_nms(boxes, scores, labels,
            ## iou_threshold); write the best max_detections into result
            ## YOUR CODE HERE
        batch['postprocessed'] = result


def average_precision(detections, boxes, labels, n_classes=10, iou_threshold=0.5):
    """Mean over classes of the VOC average precision at IoU ``iou_threshold``.

    Parameters
    ----------
    detections : dict
        ``batch['postprocessed']`` of a whole dataset: ``boxes [N, D, 4]``,
        ``scores [N, D]``, ``labels [N, D]`` (``-1`` for empty rows).
    boxes, labels : torch.Tensor
        True boxes ``[N, K, 4]`` and classes ``[N, K]`` (``-1`` for missing).

    Returns
    -------
    float
        The mean average precision over the classes that have true boxes.
    """
    precisions = []
    for class_index in range(n_classes):
        n_true = int((labels == class_index).sum())
        if n_true == 0:
            continue
        found = []
        for image_index in range(len(boxes)):
            mine = detections['labels'][image_index] == class_index
            for box, score in zip(detections['boxes'][image_index][mine],
                                  detections['scores'][image_index][mine]):
                found.append((float(score), image_index, box))
        found.sort(key=lambda item: -item[0])
        used = set()
        hits = []
        for _, image_index, box in found:
            true_boxes = boxes[image_index]
            candidates = (labels[image_index] == class_index).nonzero().flatten().tolist()
            best, best_iou = None, iou_threshold
            for candidate in candidates:
                iou = float(torchvision.ops.box_iou(box[None], true_boxes[candidate][None]))
                if iou >= best_iou and (image_index, candidate) not in used:
                    best, best_iou = candidate, iou
            hits.append(best is not None)
            if best is not None:
                used.add((image_index, best))
        true_positives = torch.tensor(hits, dtype=torch.float).cumsum(0)
        recall = torch.cat([torch.zeros(1), true_positives / n_true])
        precision = torch.cat([torch.ones(1), true_positives / torch.arange(1, len(hits) + 1)])
        # Precision envelope: the best precision at this recall or higher.
        precision = precision.flip(0).cummax(0).values.flip(0)
        precisions.append(float(((recall[1:] - recall[:-1]) * precision[1:]).sum()))
    return sum(precisions) / max(len(precisions), 1)
