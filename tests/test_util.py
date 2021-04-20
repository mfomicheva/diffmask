import unittest

import torch

from diffmask.utils.util import map_bpe_moses_bert
from diffmask.utils.metrics import confusion_matrix


class TestUtil(unittest.TestCase):

    def test_map_bpe_moses_bert(self):
        bpe_tokens = ['this', 'works', 'for', 'any', '##thing', 'any', '##where', '.']
        moses_tokens = ['this', 'works', 'for', 'anything', 'anywhere', '.']
        bpe_to_moses, moses_to_bpe = map_bpe_moses_bert(bpe_tokens, moses_tokens)
        for k, v in bpe_to_moses.items():
            print('{} --> {}'.format(bpe_tokens[k], moses_tokens[v]))
        for k, v in moses_to_bpe.items():
            for tok in v:
                print('{} --> {}'.format(moses_tokens[k], bpe_tokens[tok]))
        print(bpe_to_moses)
        print(moses_to_bpe)

    def test_confusion_matrix(self):
        y_preds = torch.zeros((8,), dtype=torch.long)
        y_true = torch.zeros((8,), dtype=torch.long)
        M = confusion_matrix(y_preds, y_true)
        assert sum(M.shape) == 4
        tn, fp, fn, tp = M.view(M.numel())
        assert tn.item() == 8
