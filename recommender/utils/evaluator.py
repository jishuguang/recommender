from collections import defaultdict
import os

import pandas as pd
from numba import njit
import numpy as np
from scipy.stats import rankdata
import torch
from torch.utils import data
from tqdm import tqdm

from utils.log import get_logger


logger = get_logger()


@njit
def _auc(actual, pred_ranks):
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fast_auc(actual, predicted):
    # https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)


class Evaluator:

    def __init__(self, evaluator_cfg, test_data, save_dir):
        self._cfg = evaluator_cfg
        self._save_dir = save_dir
        self._setup_data(test_data)

    def _setup_data(self, val_data):
        batch_size = self._cfg.device.batch_size
        num_worker = self._cfg.device.num_worker
        self._val_iter = data.DataLoader(val_data, shuffle=False,
                                         batch_size=batch_size, num_workers=num_worker,
                                         drop_last=False)

    def evaluate(self, model):
        model.eval()
        users = list()
        items = list()
        truths = list()
        preds = list()
        logger.info(f'Evaluating...')
        for val_batch in tqdm(self._val_iter):
            device = self._cfg.device.name
            for key, value in val_batch.items():
                if isinstance(value, torch.Tensor):
                    val_batch[key] = value.to(device)
            user = val_batch['userid']
            item = val_batch['itemid']
            action = val_batch['action']
            with torch.no_grad():
                pred = model.infer(val_batch)
                preds.append(pred.cpu())
                truths.append(action.cpu())
                users.append(user.cpu())
                items.append(item.cpu())

        users = torch.cat(users)
        items = torch.cat(items)
        truths = torch.cat(truths)
        preds = torch.cat(preds)
        self._dump_result(users, items, preds)

        logger.info(f'Calculating auc...')
        individual_aucs = list()
        total_auc = 0.
        total_weight = 0.
        for i, weight in enumerate(self._cfg.action_weight):
            auc = self._auc(users.numpy(), items.numpy(), truths[:, i].numpy(), preds[:, i].numpy())
            individual_aucs.append(auc)
            total_auc += auc * weight
            total_weight += weight
        return total_auc / total_weight, individual_aucs

    def _auc(self, users, items, truths, preds):
        raise NotImplementedError

    def _dump_result(self, users, items, preds):
        result = torch.cat([users.unsqueeze(1), items.unsqueeze(1), preds], dim=1)
        result_df = pd.DataFrame(result.numpy())
        result_df.columns = ['userid', 'feedid'] + self._cfg.action_name
        result_df['userid'] = result_df['userid'].astype(np.long)
        result_df['feedid'] = result_df['feedid'].astype(np.long)
        path = os.path.join(self._save_dir, 'result.csv')
        logger.info(f'Dumping result details to {path}.')
        result_df.to_csv(path, float_format='%.6f', index=False)


class UAuc(Evaluator):

    def _auc(self, users, items, truths, preds):
        """Calculate user AUC."""
        assert len(truths) == len(preds) == len(users)

        user_pred = defaultdict(lambda: [])
        user_truth = defaultdict(lambda: [])
        for idx in range(len(truths)):
            user_id = users[idx]
            pred = preds[idx]
            truth = truths[idx]
            user_pred[user_id].append(pred)
            user_truth[user_id].append(truth)

        # flag users whose samples are all positive or negative.
        user_flag = {}
        for user_id in set(users):
            cur_user_truths = user_truth[user_id]
            flag = False
            for i in range(len(cur_user_truths) - 1):
                if cur_user_truths[i] != cur_user_truths[i + 1]:
                    flag = True
                    break
            user_flag[user_id] = flag

        total_auc = 0.0
        valid_user = 0
        for user_id, flag in user_flag.items():
            if not flag:
                continue
            auc = fast_auc(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            valid_user += 1
        user_auc = float(total_auc) / valid_user
        return user_auc


EVALUATOR = {
    'uauc': UAuc
}


def build_evaluator(evaluator_cfg, val_dataset, save_dir):
    if evaluator_cfg.name in EVALUATOR:
        return EVALUATOR[evaluator_cfg.name](evaluator_cfg, val_dataset, save_dir)
    else:
        raise ValueError(f'Unsupported evaluator {evaluator_cfg.name}.')


def main():
    users = torch.tensor([1, 1]).numpy()
    items = torch.tensor([1, 2]).numpy()
    truths = torch.tensor([1, 1]).numpy()
    preds = torch.tensor([1, 0]).numpy()
    # print(UAuc._auc(users, items, truths, preds))


if __name__ == '__main__':
    main()
