import argparse

from sacremoses.tokenize import MosesTokenizer

from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile')
parser.add_argument('-o', '--outfile')
parser.add_argument('-l', '--lang')
parser.add_argument('--tokenize', default=False, action='store_true')
parser.add_argument('--min_freq', default=1, type=int)
args = parser.parse_args()

tokenizer = MosesTokenizer(lang=args.lang)
result = defaultdict(int)

print('Counting frequencies...')
for line in open(args.infile):
    line = line.strip().lower()
    tokens = tokenizer.tokenize(line, escape=False) if args.tokenize else line.split()
    for token in tokens:
        result[token] += 1

print('Writing vocabulary with counts...')
with open(args.outfile, 'w') as o:
    for token in sorted(result, key=lambda c: c[1], reverse=True):
        if result[token] < args.min_freq:
            break
        o.write('{}\t{}\n'.format(token, result[token]))
o.close()
