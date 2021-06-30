from unittest import TestCase

from torch.utils.data import DataLoader

from utils.config import get_cfg_defaults
from dataset import build_dataset
from utils.log import get_logger


logger = get_logger()


class TestDataset(TestCase):

    @classmethod
    def setUpClass(cls):
        cfg = get_cfg_defaults()
        cfg_file = r'C:\Users\gg\Documents\MySpace\git\recommender\experiments\2021_WeChat_Challenge\fm_offline.yml'
        cfg.merge_from_file(cfg_file)
        cfg.freeze()
        cls._data_cfg = cfg.data
        logger.info(f'Data config:\n{cls._data_cfg}')

    def test_data_loader(self):
        val_dataset = build_dataset(self._data_cfg, 'test')
        val_iter = DataLoader(val_dataset, shuffle=True, batch_size=2, num_workers=1, drop_last=False)
        for data in val_iter:
            print(data)
            break
