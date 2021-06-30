import argparse
import os
import time
from logging import FileHandler

from dataset import build_dataset
from model import build_model
from utils.log import get_logger
from utils.config import get_cfg_defaults
from utils.trainer import build_trainer
from utils.serialization import load_model
from utils.evaluator import build_evaluator

logger = get_logger()


def load_pretrain_model(model, pretrain_cfg):
    if pretrain_cfg is None:
        return model

    if 'resume' in pretrain_cfg:
        load_model(model, pretrain_cfg.resume, resume=True)
    elif 'load' in pretrain_cfg:
        load_model(model, pretrain_cfg.load)

    return model


def train(cfg):
    train_dataset = build_dataset(cfg.data, 'train')
    val_dataset = build_dataset(cfg.data, 'val')
    model = build_model(cfg.model)
    load_pretrain_model(model, cfg.pretrain)
    evaluator = build_evaluator(cfg.evaluator, val_dataset, cfg.save.dir)
    build_trainer(cfg.trainer, model, train_dataset, evaluator, cfg.save.dir).train()


def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--config', required=True, type=str, help='Path to config.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.save.dir = os.path.join(cfg.save.dir, time.strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(cfg.save.dir):
        os.makedirs(cfg.save.dir)
    cfg.freeze()

    logger.addHandler(FileHandler(os.path.join(cfg.save.dir, f'train.log')))
    logger.info(f'Loading config {args.config}.')
    logger.info(f'Config:\n {cfg}')
    train(cfg)


if __name__ == '__main__':
    main()
