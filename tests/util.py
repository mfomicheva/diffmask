import os
import torch
import random


torch.manual_seed(1234)
random.seed(1234)

vocab = ['es', 'una', 'frase', 'con', 'muchas', 'palabras', 'this', 'is', 'a', 'sentence', 'with', 'many', 'words']


def _make_files(data, num_examples, max_len, data_dir, file_prefix):
    offset = 0
    with open(os.path.join(data_dir, f'{file_prefix}.src'), 'w') as s:
        with open(os.path.join(data_dir, f'{file_prefix}.tgt'), 'w') as t:
            with open(os.path.join(data_dir, f'{file_prefix}.labels'), 'w') as l:
                with open(os.path.join(data_dir, f'{file_prefix}.tags'), 'w') as w:
                    for _ in range(num_examples):
                        for fh in (s, t):
                            ex_len = random.randint(1, max_len + 1)
                            print(' '.join([vocab[i] for i in data[offset:offset + ex_len]]), file=fh)
                            offset += ex_len
                        print(torch.randint(2, (1,)).item(), file=l)
                        print(' '.join(map(str, torch.randint(2, (ex_len,)).tolist())), file=w)


def create_dummy_data(data_dir, file_prefix, num_examples=100, max_len=100):
    data = torch.randint(len(vocab), (num_examples * max_len * 2,))
    _make_files(data, num_examples, max_len, data_dir, file_prefix)
