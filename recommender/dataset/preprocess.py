import numpy as np


from utils.log import get_logger


logger = get_logger()


class Preprocessor:

    @staticmethod
    def word2id(df, vocab):
        """Transform word to its id.
        :param df: DataFrame to transform.
        :param vocab: Dict{name: vocab}.
        :return: preprocessed DataFrame.
        """
        for column in df.columns:
            voc_name = column
            if voc_name not in vocab:
                continue

            logger.info(f'word2id: transforming {column}...')
            voc = vocab[voc_name]
            df[f'{column}_orig'] = df[column]
            df[column] = voc.get_id(df[column])
        return df

    @staticmethod
    def normalize(df, columns, vocab):
        """Normalize numeric feature using vocabulary id.
        :param df: DataFrame.
        :param columns: List[column].
        :param vocab: Dict{name: vocab}.
        :return: preprocessed DataFrame.
        """
        for column in columns:
            voc_name = column
            if voc_name not in vocab:
                continue

            logger.info(f'normalize {column}...')
            voc = vocab[voc_name]
            df[column] = df[column].astype(np.float) / voc.max_id()
        return df
