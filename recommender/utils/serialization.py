import os

import torch
from utils.log import get_logger

logger = get_logger()


def check_path(path):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def save_model(model, path):
    logger.info(f'Save model to {path}.')
    check_path(path)
    state_dict = {
        'state_dict': model.state_dict()
    }
    torch.save(state_dict, path)


def load_model(model, path, resume=False):
    # TODO: implement resume.
    logger.info(f'Loading model: {path}.')
    check_path(path)
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

