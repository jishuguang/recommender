import argparse
import os

import pandas as pd

from dataset.vocabulary import build_vocabulary
from utils.log import get_logger


logger = get_logger()


class IdModifier:

    def __init__(self, vocab_type, vocab_name, path):
        voc = build_vocabulary(vocab_type, vocab_name)
        voc.setup_by_csv(path)
        self._voc = voc

    def modify(self, csv_path, column, output_dir):
        logger.info(f'Reading csv {csv_path}...')
        df = pd.read_csv(csv_path)

        logger.info(f'Modifying id...')
        df[column] = self._voc.get_id(df[column])

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_name = os.path.basename(csv_path).replace('.csv', '_id_modified.csv')
        output_file = os.path.join(output_dir, output_name)

        logger.info(f'Saving to {output_file}...')
        df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Modify id using vocabulary.')
    parser.add_argument('--vocab', required=True, type=str, help='Path to vocabulary.')
    parser.add_argument('--vocab_type', required=True, choices=['cat', 'num'],
                        help='Type of the vocabulary.')
    parser.add_argument('--csv', required=True, type=str, help='Path to csv file to modify.')
    parser.add_argument('--column', required=True, type=str, help='Column to modify.')
    parser.add_argument('--output', required=False, type=str, help='Output directory, default to input directory.')
    args = parser.parse_args()

    IdModifier(args.vocab_type, 'dummy_name', args.vocab).modify(args.csv, args.column, args.output)


if __name__ == '__main__':
    main()
