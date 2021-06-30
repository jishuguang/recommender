import argparse
import os
import time

import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
import numpy as np

from model import build_model
from utils.serialization import load_model
from utils.config import get_cfg_defaults
from dataset import build_dataset
from utils.log import get_logger


logger = get_logger()


class Infer:

    def __init__(self, model_path, config_path):
        self._cfg = get_cfg_defaults()
        self._cfg.merge_from_file(config_path)
        self._cfg.freeze()
        self._save_dir = os.path.dirname(model_path)

        self._setup_model(model_path)
        self._setup_data()

    def _setup_model(self, model_path):
        self._model = build_model(self._cfg.model).to(self._cfg.evaluator.device.name)
        load_model(self._model, model_path)

    def _setup_data(self):
        test_data = build_dataset(self._cfg.data, 'test')
        self._data_name = os.path.basename(self._cfg.data.action.test).split('.')[0]
        batch_size = self._cfg.evaluator.device.batch_size
        num_worker = self._cfg.evaluator.device.num_worker
        self._test_iter = data.DataLoader(test_data, shuffle=False,
                                          batch_size=batch_size, num_workers=num_worker,
                                          drop_last=False)

    def infer(self):
        self._model.eval()
        users = list()
        items = list()
        preds = list()
        logger.info(f'Infering...')
        for batch in tqdm(self._test_iter):
            device = self._cfg.evaluator.device.name
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            user = batch['userid']
            item = batch['itemid']
            with torch.no_grad():
                pred = self._model.infer(batch)
                preds.append(pred.cpu())
                users.append(user.cpu())
                items.append(item.cpu())

        users = torch.cat(users)
        items = torch.cat(items)
        preds = torch.cat(preds)
        self._dump_result(users, items, preds)

    def _dump_result(self, users, items, preds):
        result = torch.cat([users.unsqueeze(1), items.unsqueeze(1), preds], dim=1)
        result_df = pd.DataFrame(result.numpy())
        result_df.columns = ['userid', 'feedid'] + self._cfg.evaluator.action_name
        result_df['userid'] = result_df['userid'].astype(np.long)
        result_df['feedid'] = result_df['feedid'].astype(np.long)
        path = os.path.join(self._save_dir, f'{self._data_name}_{time.strftime("%Y%m%d%H%M%S")}.csv')
        logger.info(f'Dumping inferred result to {path}.')
        result_df.to_csv(path, float_format='%.6f', index=False)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model.')
    parser.add_argument('--model', required=True, type=str, help='The path to model.')
    parser.add_argument('--config', required=True, type=str, help='The path to config.')
    args = parser.parse_args()

    Infer(args.model, args.config).infer()


if __name__ == '__main__':
    main()
