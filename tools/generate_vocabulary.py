import argparse
import os

import pandas as pd

from utils.config import get_cfg_defaults
from dataset.vocabulary import CategoricalVocabulary, NumericVocabulary
from utils.log import get_logger


logger = get_logger()


class VocabularyGenerator:

    TYPES = ('cat', 'num')

    def __init__(self, output):
        if not os.path.exists(output):
            os.mkdir(output)
        self._output_path = output
        self._names = {t: list() for t in self.TYPES}
        self._dims = {t: list() for t in self.TYPES}

    def _load_csv(self, csv_file):
        if not os.path.exists(csv_file):
            raise ValueError(f'Input csv file does not exist: {csv_file}.')
        logger.info(f'Loading csv: {csv_file}...')
        df_input = pd.read_csv(csv_file)
        return df_input

    def generate(self, vocabs):
        """Generate vocabularies.
        :param vocabs:  List[
                            Dict{
                                input: path
                                vocabs: List[Dict{
                                                type: cat/num
                                                name: List[column]
                                                paras: ...
                                            }
                                        ]
                            }
                        ]
        :return: None.
        """
        for vocab_cfg in vocabs:
            df_input = self._load_csv(vocab_cfg['input'])
            for vocab in vocab_cfg['vocabs']:
                if vocab['type'] == 'cat':
                    self._generate_cat_vocabulary(df_input, vocab)
                elif vocab['type'] == 'num':
                    self._generate_num_vocabulary(df_input, vocab)
                else:
                    raise ValueError(f'Unsupported vocab type: {vocab["type"]}.')

        for t in self.TYPES:
            logger.info(f'{t}:\n'
                        f'names: {self._names[t]}\n'
                        f'dims: {self._dims[t]}')

    def _generate_cat_vocabulary(self, df_input, cat_vocab_cfg):
        logger.info(f'Generating categorical vocabulary...')

        cat_name = cat_vocab_cfg['name']
        cat_min_frequency = cat_vocab_cfg['min_frequency']
        if isinstance(cat_min_frequency, int):
            cat_min_frequency = [cat_min_frequency] * len(cat_name)

        for name, min_frequency in zip(cat_name, cat_min_frequency):
            if name not in df_input.columns:
                logger.info(f'Category {name} is not in input csv, skip.')
                continue

            voc_name = f'{name}_cat'
            logger.info(f'Generating {voc_name} vocabulary...')
            vocab = CategoricalVocabulary(voc_name)
            vocab.setup_by_word(df_input[name], min_frequency=min_frequency)
            file_path = os.path.join(self._output_path, f'{voc_name}.csv')
            vocab.save(file_path)

            self._names['cat'].append(name)
            self._dims['cat'].append(len(vocab))

    def _generate_num_vocabulary(self, df_input, num_vocab_cfg):
        logger.info(f'Generating numeric vocabulary...')

        num_name = num_vocab_cfg['name']
        num_bucket = num_vocab_cfg['bucket']
        if isinstance(num_bucket, int):
            num_bucket = [num_bucket] * len(num_name)
        num_sparse_positive = num_vocab_cfg['sparse_positive']
        if isinstance(num_sparse_positive, bool):
            num_sparse_positive = [num_sparse_positive] * len(num_name)

        for name, bucket, sparse_positive in zip(num_name, num_bucket, num_sparse_positive):
            if name not in df_input.columns:
                logger.info(f'Numeric feature {name} is not in input csv, skip.')
                continue

            voc_name = f'{name}_num'
            logger.info(f'Generating {voc_name} vocabulary...')
            vocab = NumericVocabulary(voc_name)
            vocab.setup_by_word(df_input[name], buckets=bucket, sparse_positive=sparse_positive)
            file_path = os.path.join(self._output_path, f'{voc_name}.csv')
            vocab.save(file_path)

            self._names['num'].append(name)
            self._dims['num'].append(len(vocab))


def main():
    parser = argparse.ArgumentParser(description='Generate vocabulary.')
    parser.add_argument('--config', required=True, type=str, help='Path to vocabulary config.')
    parser.add_argument('--output', required=True, type=str, help='Output path.')
    args = parser.parse_args()

    # load config
    logger.info(f'Loading config {args.config}...')
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    VocabularyGenerator(args.output).generate(cfg.vocabs)


if __name__ == '__main__':
    main()
