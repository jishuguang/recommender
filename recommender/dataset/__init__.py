from .point_dataset import build_point_dataset
from .pair_dataset import build_pair_dataset
from utils.log import get_logger


logger = get_logger()


__all__ = ['build_dataset']


DATASETS = {
    'point': build_point_dataset,
    'pair': build_pair_dataset
}

PURPOSE = {'train', 'val', 'test'}


def build_dataset(data_cfg, purpose):
    if purpose not in PURPOSE:
        raise ValueError(f'Argument purpose should be one of {PURPOSE}')
    if data_cfg.name in DATASETS:
        logger.info(f'Dataset is set to \"{data_cfg.name}\" form for {purpose}.')
        return DATASETS[data_cfg.name](data_cfg, purpose)
    else:
        raise ValueError(f'Unsupported data format：{data_cfg.name}.')
