import unittest
import os
import tempfile

from transformers import XLMRobertaTokenizer

from diffmask.models.quality_estimation import load_sent_level
from tests.util import create_dummy_data


class TestQualityEstimationModel(unittest.TestCase):

    def test_prepare_data(self):
        bert_model = "xlm-roberta-base"
        tokenizer = XLMRobertaTokenizer.from_pretrained(bert_model)
        file_prefix = 'test'
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            create_dummy_data(data_dir, file_prefix)
            dataset = load_sent_level(
                os.path.join(data_dir, f'{file_prefix}.src'),
                os.path.join(data_dir, f'{file_prefix}.tgt'),
                os.path.join(data_dir, f'{file_prefix}.labels'),
                tokenizer,
                path_word_labels=os.path.join(data_dir, f'{file_prefix}.tags')
            )
            assert dataset[0][1][0].shape[0] == 128
            assert sum(dataset[0][1][1]).item() == 20
            assert dataset[0][1][2].item() == 0
