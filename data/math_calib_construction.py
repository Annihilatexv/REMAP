import json
import fire
import random
import os.path as osp
import pandas as pd
from tqdm import tqdm


def main(dataset_path: str = './data/train-00000-of-00001-7320a6f3aba8ebd2.parquet', output_path: str = './', seed: int = 42):
    random.seed(seed)
    data_all = []
    
    # Read parquet file
    df = pd.read_parquet(dataset_path)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Assume the columns in the parquet are 'problem' and 'solution'
        it = str(row['problem']) + ' ' + str(row['solution'])
        data_all.append({'text': it})

    random.shuffle(data_all)
    with open(osp.join(output_path, 'math_pretrain_style.json'), 'w') as f:
        json.dump(data_all, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    fire.Fire(main)
