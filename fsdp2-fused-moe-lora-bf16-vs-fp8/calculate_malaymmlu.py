from glob import glob
import json
import re
import os
import click
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

def aggregate_mcq_predictions(preds, gold, method="any"):
    if method == "majority":
        guess = Counter(preds).most_common(1)[0][0]

    elif method == "any":
        return int(gold in preds)

    else:
        raise ValueError("method must be 'majority' or 'any'")

    return int(guess == gold)
    
def parse(text):
    r = re.findall(r"\b[A-F]\b", text)
    return r[0]

with open('MalayMMLU_0shot.json') as fopen:
    malaymmlu = json.load(fopen)

@click.command()
@click.option("--pattern", default='malaymmlu-ds3-qwen2.5-72b-lora-256-checkpoint-*', help="malaymmlu glob pattern")
def main(pattern):
    folders = glob(pattern)
    folders = [f for f in folders if '.zip' not in f and 'baseline' not in f and '.ipynb' not in f]
    for f in folders:
        print(f, len(glob(os.path.join(f, '*.json'))))
    print()

    for f in folders:
        total_k1 = defaultdict(int)
        total_k5 = defaultdict(int)
        total = defaultdict(int)
        wrong = []
        for i in range(len(malaymmlu)):
            try:
                results = []
                for k in range(3):
                    try:
                        filename = os.path.join(f, f'{i}-{k}.json')
                        with open(filename) as fopen:
                            d = json.load(fopen)
                        p = parse(d)
                        if p:
                            results.append(p)
                    except:
                        pass
                if len(results):
                    s = aggregate_mcq_predictions(results[:3], malaymmlu[i]['key'])
                    total_k1[malaymmlu[i]['category']] += aggregate_mcq_predictions(results[:1], malaymmlu[i]['key'])
                    total_k5[malaymmlu[i]['category']] += s
        
                    if s == 0:
                        wrong.append((results, malaymmlu[i]))
            except Exception as e:
                pass
            
            total[malaymmlu[i]['category']] += 1
        
        print(f)
        avg_k1 = []
        avg_k5 = []
        for k, v in total.items():
            print(k, total_k1[k] / v, total_k5[k] / v)
            avg_k1.append(total_k1[k] / v)
            avg_k5.append(total_k5[k] / v)
        print('average', np.mean(avg_k1), np.mean(avg_k5))

        print()

if __name__ == "__main__":
    main()