from unittest import TestCase

import torch

from model import build_model
from utils.config import get_cfg_defaults


class TestDataset(TestCase):

    @classmethod
    def setUpClass(cls):
        cfg = get_cfg_defaults()
        cfg_file = r'C:\Users\gg\Documents\MySpace\git\recommender\experiments\2021_WeChat_Challenge\fm_offline.yml'
        cfg.merge_from_file(cfg_file)
        cfg.freeze()
        cls._model_cfg = cfg.model

    def test_fm(self):
        device = 'cuda:0'
        fm = build_model(self._model_cfg).to(device)
        dummy_data = {
            'cat': torch.randn(2, 21).to(device).to(torch.int),
            'context': torch.randint(10, [2, 2]).to(device).to(torch.int)
        }
        fm(dummy_data)
