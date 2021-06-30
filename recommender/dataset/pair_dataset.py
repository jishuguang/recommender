import pandas as pd
import random

from .base_dataset import BaseDataset, SupervisedBaseDataset
from utils.log import get_logger


logger = get_logger()


class PairTrainDataset(SupervisedBaseDataset):

    def _setup(self):
        super()._setup()

        user_column = self._cfg.input.userid
        action_name = self._cfg.input.action

        def _agg_user(user_df):
            """Aggregate positive indexes, start index and end index of a user."""
            user_info = {'start': user_df.index[0], 'end': user_df.index[-1]}
            length = user_info['end'] - user_info['start'] + 1
            valid_amount = 0
            for action in action_name:
                positive_indexes = user_df.index[user_df[action] == 1].values
                action_valid_amount = len(positive_indexes)
                if 0 < action_valid_amount < length:
                    positive_info = ' '.join(str(n) for n in positive_indexes)
                    valid_amount += action_valid_amount
                else:
                    # all records are positive or negative
                    positive_info = False
                user_info[action + '_positive'] = positive_info
            user_info['valid_amount'] = valid_amount
            return pd.Series(user_info)

        # sort action df based on userid in order to group records of each user.
        logger.info('Sorting action data...')
        self._df.sort_values(by=[user_column], inplace=True)
        self._df.reset_index(drop=True, inplace=True)

        # aggregate info of each user, including:
        # xxx_positive: positive indexes str for each action, i.e., "i1,i2..."
        #               (if the action for this user is invalid, set it to False);
        # start: original start index;
        # end: original end index;
        # valid_amount: sum of valid positives of actions
        # index: new index for __getitem__
        user_group = self._df[[user_column] + action_name].groupby([user_column])
        self._user_agg_info = user_group.apply(_agg_user).reset_index()
        self._user_agg_info['index'] = self._user_agg_info['valid_amount'].cumsum() - 1

        self._len = self._user_agg_info['valid_amount'].sum()
        logger.info(f'Adjust dataset size to {len(self)}.')

    def __getitem__(self, index):
        """
        Return a pair of (positive, negative).
        A user with any positive action is treated at least as one positive,
        if there are more than one positive action in the single record,
        each positive action will be treated individually with different negative
        sampling.
        """
        index_sr = self._user_agg_info['index']
        info_index = self._user_agg_info.index[index_sr >= index].min()
        cur_info = self._user_agg_info.iloc[info_index]
        end_index = cur_info['index']

        # get original index in action_df for current index
        positive_indexes = None
        positive_index = None
        for action in reversed(self._cfg.input.action):
            key = action + '_positive'
            if not cur_info[key]:
                continue

            positive_indexes = [int(i) for i in cur_info[key].split()]
            end_index -= len(positive_indexes)
            if end_index >= index:
                continue

            positive_index = positive_indexes[index - end_index - 1]
            break

        if positive_indexes is None and positive_index is None:
            raise ValueError(f'Current Info {cur_info}.')

        # negative sampling
        negative_indexes = list(set(range(cur_info['start'], cur_info['end'] + 1)) - set(positive_indexes))
        negative_index = random.choice(negative_indexes)

        data = {
            'positive': super().__getitem__(positive_index),
            'negative': super().__getitem__(negative_index)
        }
        return data


DATASETS = {
    'train': PairTrainDataset,
    'val': SupervisedBaseDataset,
    'test': BaseDataset
}


def build_pair_dataset(data_cfg, purpose='train'):
    if purpose in DATASETS:
        return DATASETS[purpose](data_cfg, purpose)
    else:
        raise ValueError(f'Unsupported purpose {purpose}.')
