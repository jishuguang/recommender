import os
import glob

from torch.utils.data import Dataset
import pandas as pd
import torch

from .vocabulary import CategoricalVocabulary, NumericVocabulary
from .preprocess import Preprocessor
from utils.log import get_logger

logger = get_logger()


class BaseDataset(Dataset):
    """Base class for all datasets."""

    DATA_NAME = ('user', 'item', 'action')
    # TODO: support multi_hot feature.
    DATA_TYPE = {
        'userid': torch.int64,
        'itemid': torch.int64,
        'num': torch.float32,
        'cat': torch.int64,  # one_hot
        'multi_hot': torch.tensor,
        'context': torch.int64,
        'action': torch.float32,
    }

    def __init__(self, data_cfg, purpose):
        super().__init__()
        self._cfg = data_cfg
        self._purpose = purpose
        self._setup()

    def _setup(self):
        """Setup dataset.."""
        self._vocabulary = self._setup_vocabulary()
        self._df = self._setup_df()
        self._data = self._setup_data()
        self._len = len(self._df)
        logger.info(f'Action dataset total size: {len(self)}.')

    def _setup_vocabulary(self):
        """Setup vocabulary.
        :return: dict{name: vocab}.
        """
        vocabulary = dict()
        if 'vocab' not in self._cfg:
            logger.info(f'Config \"vocab\" is not set, skip setup vocabulary.')
            return vocabulary

        path = self._cfg.vocab
        if os.path.isfile(path):
            vocab_files = [path]
        else:
            vocab_files = list(glob.glob(os.path.join(path, '*.csv')))

        for file in vocab_files:
            # file_name: "voc_name + type"
            file_name = os.path.basename(file).replace('.csv', '')
            voc_name, voc_type = file_name.rsplit('_', 1)
            # TODO: get vocab_type in a better way, make file_name and vocab_type independent.
            voc_cls = CategoricalVocabulary if voc_type == 'cat' else NumericVocabulary
            voc = voc_cls(voc_name)
            voc.setup_by_csv(file)
            vocabulary[voc_name] = voc
        return vocabulary

    def _setup_df(self):
        """Setup DataFrame.
        :return: pd.DataFrame.
        """
        df_dict = dict()
        for data_name in self.DATA_NAME:
            # action is required
            df_dict[data_name] = self._setup_individual_df(data_name, data_name == 'action')

        logger.info(f'Joining data for {self._purpose}...')
        df = self._join_data(df_dict['action'], df_dict['user'], 'user')
        df = self._join_data(df, df_dict['item'], 'item')
        return df

    def _setup_individual_df(self, data_name, required=False):
        """Setup individual DataFrame.
        :param data_name: the name of DataFrame.
        :return: pd.DataFrame or None.
        """
        if data_name not in self._cfg:
            if required:
                raise ValueError(f'Data \"{data_name}\" is not set.')
            logger.info(f'Data \"{data_name}\" is not set, skipping setup {data_name}.')
            return None

        # load data
        path = self._cfg[data_name]
        if data_name == 'action':
            path = self._cfg[data_name][self._purpose]
        logger.info(f'Loading {data_name} data: {path}...')
        df = pd.read_csv(path)

        # preprocess
        # TODO: for now, "word2id + normalize" preprocessing pipeline is hard coded
        df = Preprocessor.word2id(df, self._vocabulary)

        # TODO: ugly code.
        input_cfg = self._cfg.input
        num_input = getattr(input_cfg, 'num', None)
        if num_input is not None:
            if getattr(num_input, data_name, None) is not None:
                df = Preprocessor.normalize(df, num_input[data_name], self._vocabulary)
        return df

    def _join_data(self, df, df_another, data_name):
        """
        Join df_another to df.
        :param df: DataFrame to join.
        :param df_another: another DataFrame.
        :param data_name: name of another DataFrame.
        :return: merged DataFrame.
        """
        if df_another is None:
            return df

        # collect columns for joining.
        input_cfg = self._cfg.input
        columns = list()
        for field in ('num', 'cat', 'multi_hot'):
            if field not in input_cfg or data_name not in input_cfg[field]:
                continue
            columns.extend(input_cfg[field][data_name])
        if not columns:
            return df

        # join
        logger.info(f'Join {data_name} to action: {columns}.')
        data_id = input_cfg[f'{data_name}id']
        df_to_join = df_another[[data_id] + columns]
        return df.join(df_to_join, on=[data_id], how='left')

    def _setup_data(self):
        """Setup data based on self._df.
        :return: dict.
        """
        data = dict()
        input_cfg = self._cfg.input

        # userid, itemid
        for field in ('userid', 'itemid'):
            data[field] = torch.tensor(self._df[input_cfg[field]].values, dtype=self.DATA_TYPE[field])

        # num/cat/multi_hot
        for field in ('num', 'cat', 'multi_hot'):
            if field not in input_cfg:
                continue
            # collect columns of this field
            columns = list()
            for data_name in self.DATA_NAME:
                if data_name in input_cfg[field]:
                    columns.extend(input_cfg[field][data_name])
            data[field] = torch.tensor(self._df[columns].values, dtype=self.DATA_TYPE[field])

        # context
        field = 'context'
        if field in input_cfg:
            columns = input_cfg[field]
            data[field] = torch.tensor(self._df[columns].values, dtype=self.DATA_TYPE[field])

        return data

    def __getitem__(self, index):
        return {key: value[index] for key, value in self._data.items()}

    def __len__(self):
        return self._len


class SupervisedBaseDataset(BaseDataset):

    def _setup_data(self):
        data = super()._setup_data()
        # setup action data
        field = 'action'
        action_columns = self._cfg.input[field]
        data[field] = torch.tensor(self._df[action_columns].values, dtype=self.DATA_TYPE[field])
        return data
