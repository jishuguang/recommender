import os

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data

from utils.loss import build_loss
from utils.log import get_logger
from utils.serialization import save_model

logger = get_logger()


class Trainer:

    def __init__(self, train_cfg, model, train_data, evaluator, save_dir):
        self._cfg = train_cfg
        self._model = model.to(self._cfg.device.name)
        self._setup_data(train_data)
        self._setup_scheduler()
        self._loss = build_loss(self._cfg.loss)
        self._evaluator = evaluator
        self._save_dir = save_dir

    def _setup_data(self, train_data):
        batch_size = self._cfg.device.batch_size
        num_worker = self._cfg.device.num_worker
        logger.info(f'batch_size: {batch_size}')
        logger.info(f'num_workers: {num_worker}')
        self._total_steps = len(train_data) // batch_size
        self._train_iter = data.DataLoader(train_data, shuffle=True,
                                           batch_size=batch_size, num_workers=num_worker,
                                           drop_last=True)

    def _setup_scheduler(self):
        learn_paras = {
            'lr': self._cfg.learn.lr,
            'weight_decay': self._cfg.learn.weight_decay
        }
        self._optimizer = getattr(optim, self._cfg.learn.method)(self._model.parameters(), **learn_paras)
        self._scheduler = MultiStepLR(self._optimizer, self._cfg.learn.milestones)

    def _calculate_loss(self, args_list):
        raise NotImplementedError

    def train(self):
        best_metric = 0
        for epoch in range(self._cfg.learn.epochs):
            self._train_one_epoch(epoch)
            if (epoch + 1) % self._cfg.val_interval == 0:
                metric, individual_metric = self._evaluator.evaluate(self._model)
                metric_msg = f'Epoch {epoch}: metric {metric:.6f} | ' \
                    + ' | '.join([f'{action} {individual_metric[i]:.6f}'
                                  for i, action in enumerate(self._cfg.action_name)])
                logger.info(metric_msg)
                if metric > best_metric:
                    best_metric = metric
                    save_model(self._model, os.path.join(self._save_dir, 'model', 'model_best.pth'))
                    with open(os.path.join(self._save_dir, 'model', 'evaluation.txt'), 'a') as f:
                        f.write(metric_msg + os.linesep)
            save_model(self._model, os.path.join(self._save_dir, 'model', 'model_last.pth'))
            self._scheduler.step()

    def _train_one_epoch(self, epoch):
        step = 0
        self._model.train()
        for train_batch in self._train_iter:
            self._optimizer.zero_grad()
            loss, individual_loss = self._calculate_loss(train_batch)
            logger.info(f'[Epoch {epoch}][Step {step}/{self._total_steps}] '
                        f'lr: {self._scheduler.get_last_lr()[0]:.5f} | '
                        f'total_loss {loss:.4f} | '
                        + ' | '.join([f'{action} {individual_loss[i]:.4f}'
                                      for i, action in enumerate(self._cfg.action_name)]))
            loss.backward()
            self._optimizer.step()
            step += 1

    def _to_device(self, train_batch):
        """Move a batch to specified device."""
        device = self._cfg.device.name
        for key, value in train_batch.items():
            if isinstance(value, torch.Tensor):
                train_batch[key] = value.to(device)


class PointTrainer(Trainer):

    def _calculate_loss(self, train_batch):
        self._to_device(train_batch)
        logits = self._model(train_batch)
        return self._loss.calculate_loss(logits, train_batch['action'])


class PairTrainer(Trainer):

    def _calculate_loss(self, train_batch):
        logit_pair = list()
        action_pair = list()
        for sample in ('positive', 'negative'):
            batch = train_batch[sample]
            self._to_device(batch)
            logit_pair.append(self._model(batch))
            action_pair.append(batch['action'])
        return self._loss.calculate_loss(logit_pair, action_pair)


TRAINER = {
    'point': PointTrainer,
    'pair': PairTrainer
}


def build_trainer(trainer_cfg, *args):
    if trainer_cfg.name in TRAINER:
        return TRAINER[trainer_cfg.name](trainer_cfg, *args)
    else:
        raise ValueError(f'Unsupported loss {trainer_cfg.name}.')
