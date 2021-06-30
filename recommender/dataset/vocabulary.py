import pandas as pd
import os
import math

import numpy as np

from utils.log import get_logger


logger = get_logger()


class Vocabulary:
    """Base class for all vocabulary."""

    def __init__(self, name, **kwargs):
        self._name = name
        self._id_name = kwargs.get('id_name', 'id')
        self._word_name = kwargs.get('word_name', 'word')
        self._default_value = kwargs.get('default', '<default>')
        self._df = None

    def setup_by_csv(self, path):
        """Setup vocabulary using csv.
        :param path: path to csv file or its parent path.
        :return: None.
        """
        file_path = path
        if os.path.isdir(path):
            file_path = os.path.join(path, f'{self._name}.csv')
        if not os.path.exists(file_path):
            raise FileExistsError(f'Vocabulary file {file_path} does not exist.')

        logger.info(f'Setup vocabulary {self._name} with {file_path}.')
        df = pd.read_csv(file_path)
        df.columns = [self._id_name, self._word_name]

        self._df = df
        logger.info(f'Setup {self._name} vocabulary done, size: {len(self)}')

    def setup_by_word(self, words, **kwargs):
        """Setup vocabulary using words.
        :param words: a list of words or a series of words.
        :return: None.
        """
        if not (isinstance(words, pd.Series) or isinstance(words, list)):
            raise ValueError(f'words should be a list or a pd.Series.')

        min_frequency = kwargs.get('min_frequency', 5)
        # filter words whose frequency < min_frequency
        df = pd.DataFrame(words).reset_index()
        df.columns = [self._id_name, self._word_name]
        df_count = df.groupby(self._word_name).count().reset_index()
        word_sr = df_count[self._word_name][df_count[self._id_name] >= min_frequency]

        # add default value at the beginning
        word_list = [self._default_value] + list(word_sr)

        df = pd.DataFrame(word_list).reset_index()
        df.columns = [self._id_name, self._word_name]
        self._df = df
        logger.info(f'Setup {self._name} vocabulary done, size: {len(self)}')

    def __getitem__(self, index):
        """Get items using index.
        :param index: a list of indexes or a series of indexes.
        :return: a array of words.
        """
        if not (isinstance(index, list) or isinstance(index, pd.Series)):
            raise ValueError(f'index should be a int list or a int Series.')

        return self._df.set_index(self._id_name).loc[index][self._word_name].values

    def get_id(self, words):
        """Get ids using words.
        :param words: a list of words or a series of words.
        :return: a array of ids.
        """
        if not (isinstance(words, list) or isinstance(words, pd.Series)):
            raise ValueError(f'words should be a list or a Series.')

        word_sr = pd.Series(words).astype(str)
        word_sr[~word_sr.isin(self._df[self._word_name])] = self._default_value
        return self._df.set_index(self._word_name).loc[word_sr][self._id_name].values

    def __len__(self):
        if self._df is None:
            return 0
        return self._df.shape[0]

    def save(self, path):
        """Save vocabulary to path.
        :param path: csv file path or its parent path.
        :return: None.
        """
        if self._df is None:
            raise ValueError(f'Can not save: {self._name} vocabulary dataframe is None.')

        file_path = path
        if os.path.isdir(path):
            file_path = os.path.join(path, f'{self._name}.csv')

        self._df.to_csv(file_path, index=False)
        logger.info(f'Save vocabulary {self._name} to {file_path}.')

    def max_id(self):
        """Return max id."""
        return self._df[self._id_name].max()


class CategoricalVocabulary(Vocabulary):
    """Vocabulary appropriate for categorical features."""

    def setup_by_csv(self, path):
        super().setup_by_csv(path)
        # change the type of word column to str
        self._df[self._word_name] = self._df[self._word_name].astype(str)

    def setup_by_word(self, words, **kwargs):
        super().setup_by_word(words, **kwargs)
        # change the type of word column to str
        self._df[self._word_name] = self._df[self._word_name].astype(str)


class NumericVocabulary(Vocabulary):
    """Vocabulary appropriate for numeric features."""

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._word_name = kwargs.get('word_name', 'value')

    def setup_by_word(self, words, **kwargs):
        """Setup vocabulary using words, i.e., values.
        :param words: a list of values or a series of values.
        :return: None.
        """
        if not (isinstance(words, pd.Series) or isinstance(words, list)):
            raise ValueError(f'words should be a list or a pd.Series.')

        buckets = max(kwargs.get('buckets', 10), 2)
        # value >= 0 and most of the values are zeros
        sparse_positive = kwargs.get('sparse_positive', False)

        # calculate word_list [quantile_1, ..., math.inf]
        # first bucket: (-math.inf, quantile_1)
        # last bucket: [quantile_n, math.inf)
        word_sr = pd.Series(words)
        if sparse_positive:
            # remove zeros
            word_sr = word_sr[word_sr > 0]
        word_list = list(word_sr.quantile(np.linspace(0, 1, buckets + 1)).iloc[1:-1])
        word_list.append(math.inf)
        if sparse_positive:
            # insert a tiny value at the beginning
            word_list = [1e-8] + word_list

        df = pd.DataFrame(word_list).reset_index()
        df.columns = [self._id_name, self._word_name]
        self._df = df
        logger.info(f'Setup {self._name} vocabulary done, size: {len(self)}')

    def get_id(self, words):
        """Get ids using words.
        :param words: a list of words or a series of words.
        :return: a array of ids.
        """
        if not (isinstance(words, list) or isinstance(words, pd.Series)):
            raise ValueError(f'words should be a list or a Series.')

        word_sr = pd.Series(words)
        # initial index is -1
        id_sr = pd.Series(np.ones(words.shape, dtype=np.int) * -1)
        for index, value in self._df.values:
            id_sr[(word_sr < value) & (id_sr == -1)] = index

        return id_sr.values


class MultiHotVocabulary:
    """Vocabulary appropriate for multi-hot features."""
    # TODO: implement this class.
    pass


VOCABS = {
    'cat': CategoricalVocabulary,
    'num': NumericVocabulary,
    'multi_hot': MultiHotVocabulary
}


def build_vocabulary(voc_type, *args, **kwargs):
    if voc_type in VOCABS:
        return VOCABS[voc_type](*args, **kwargs)
    else:
        raise ValueError(f'Invalid vocabulary type: {voc_type}.')
