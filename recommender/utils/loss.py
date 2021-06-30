import torch
import torch.nn.functional as F

from utils.log import get_logger


logger = get_logger()


class FocalBceLoss:
    """Focal binary cross entropy."""

    def __init__(self, loss_cfg):
        self._cfg = loss_cfg
        self._beta = loss_cfg.beta
        self._weights = torch.tensor(loss_cfg.action_weight)

    def calculate_loss(self, logits, truths):
        loss = F.binary_cross_entropy_with_logits(logits, truths, reduction='none')

        logit_sigmoid = logits.sigmoid()
        scale_factor = torch.where(truths == 1, 1 - logit_sigmoid, logit_sigmoid) ** self._beta
        loss = loss * scale_factor

        individual_loss = loss.mean(dim=0)

        weights = self._weights.to(loss.device)
        mean_loss = (individual_loss * weights).sum() / weights.sum()
        return mean_loss, individual_loss


class FocalRankHingeLoss:
    """Focal rank hinge loss."""

    def __init__(self, loss_cfg):
        self._cfg = loss_cfg
        self._beta = self._cfg.beta
        assert self._beta > 0
        self._weights = torch.tensor(self._cfg.action_weight)

    def calculate_loss(self, logit_pair, truth_pair):
        distances = logit_pair[0].sigmoid() - logit_pair[1].sigmoid()
        real_distances = truth_pair[0] - truth_pair[1]
        truths = torch.abs(real_distances)

        loss = (real_distances - distances) * truths
        loss = loss ** self._beta

        valid_count = truths.sum(dim=0) + 1e-8
        individual_loss = loss.sum(dim=0) / valid_count

        weights = torch.where(valid_count == 0, 0, self._weights.to(loss.device))
        mean_loss = (individual_loss * weights).sum() / weights.sum()
        return mean_loss, individual_loss


class FocalBprLoss:
    """Focal bayesian personalized ranking Loss."""

    def __init__(self, loss_cfg):
        self._cfg = loss_cfg
        self._beta = self._cfg.beta
        assert self._beta > 0
        self._weights = torch.tensor(self._cfg.action_weight)

    def calculate_loss(self, logit_pair, truth_pair):
        distances = logit_pair[0] - logit_pair[1]
        real_distances = truth_pair[0] - truth_pair[1]
        truths = torch.abs(real_distances)
        distances = distances * real_distances

        loss = F.binary_cross_entropy_with_logits(distances, truths, reduction='none') * truths
        loss = loss ** self._beta

        valid_count = truths.sum(dim=0) + 1e-8
        individual_loss = loss.sum(dim=0) / valid_count

        weights = torch.where(valid_count == 0, 0, self._weights.to(loss.device))
        mean_loss = (individual_loss * weights).sum() / weights.sum()
        return mean_loss, individual_loss


LOSS = {
    'bce': FocalBceLoss,
    'rank_hinge': FocalRankHingeLoss,
    'bpr': FocalBprLoss
}


def build_loss(loss_cfg):
    if loss_cfg.name in LOSS:
        return LOSS[loss_cfg.name](loss_cfg)
    else:
        raise ValueError(f'Unsupported loss {loss_cfg.name}.')
