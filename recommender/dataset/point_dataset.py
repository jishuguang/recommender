import random

from .base_dataset import BaseDataset, SupervisedBaseDataset
from utils.log import get_logger


logger = get_logger()


class PointTrainDataset(SupervisedBaseDataset):

    def _setup(self):
        super()._setup()

        total_amount = len(self)

        # if any action is true, we treat it as positive.
        self._positive_indexes = list(self._df.index[self._df[self._cfg.input.action].agg('sum', axis=1) > 0])
        self._negative_indexes = list(set(range(total_amount)) - set(self._positive_indexes))
        self._positive_amount = len(self._positive_indexes)

        # adjust data_size = positive + negative_sample
        self._len = int(min(self._positive_amount * (1 + self._cfg.negative_sample), total_amount))
        logger.info(f'Positive sample size: {self._positive_amount}; '
                    f'Negative sample size: {total_amount - self._positive_amount}')
        logger.info(f'Adjust dataset size to {len(self)}')

    def __getitem__(self, index):
        if index < self._positive_amount:
            real_index = self._positive_indexes[index]
        else:
            real_index = random.choice(self._negative_indexes)
        return super().__getitem__(real_index)


DATASETS = {
    'train': PointTrainDataset,
    'val': SupervisedBaseDataset,
    'test': BaseDataset
}


def build_point_dataset(data_cfg, purpose='train'):
    if purpose in DATASETS:
        return DATASETS[purpose](data_cfg, purpose)
    else:
        raise ValueError(f'Unsupported purpose {purpose}.')
