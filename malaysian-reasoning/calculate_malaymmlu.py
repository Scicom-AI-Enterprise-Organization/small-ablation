from glob import glob
import json
import re
import os
from tqdm import tqdm
from collections import Counter

def aggregate_mcq_predictions(preds, gold, method="any"):
    if method == "majority":
        guess = Counter(preds).most_common(1)[0][0]

    elif method == "any":
        return int(gold in preds)

    else:
        raise ValueError("method must be 'majority' or 'any'")

    return int(guess == gold)
    
def parse(text):

    m = re.search(r"[A-F]", text)
    if m:
        return m.group()

with open('MalayMMLU_0shot.json') as fopen:
    malaymmlu = json.load(fopen)

folders = glob('malaymmlu-malaysian-reasoning-20b*')
folders = [f for f in folders if '.zip' not in f and 'baseline' not in f and '.ipynb' not in f]
for f in folders:
    print(f, len(glob(os.path.join(f, '*.json'))))

for f in folders:
    total_k1 = 0
    total_k5 = 0
    total = 0
    wrong = []
    for i in range(len(malaymmlu)):
        try:
            results = []
            for k in range(5):
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
                total_k1 += aggregate_mcq_predictions(results[:1], malaymmlu[i]['key'])
                total_k5 += s
                total += 1
    
                if s == 0:
                    wrong.append((results, malaymmlu[i]))
        except Exception as e:
            pass

    if total > 0:
        print(f, total_k1 / total, total_k5 / total)