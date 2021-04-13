import os
import unittest
import tempfile

from diffmask.attributions.attribute_qe import AttributeQE
from diffmask.utils.getter_setter import roberta_getter, roberta_setter
from diffmask.utils.evaluate_qe import EvaluateQE

from tests.util import make_hparams, train_model


class TestAttributeQE(unittest.TestCase):

    def test_attribute_and_evaluate_qe_roberta(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = make_hparams(data_dir, 'roberta', 'xlm-roberta-base')
            qe = train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)
            attributor = AttributeQE(qe, roberta_getter, roberta_setter, range(14), "cpu")
            attributor.make_attributions(steps=1)
            data = attributor.make_data()
            assert len(data) == 2
            example = data[0]
            layer_id = 0
            assert example[0].attributions_target[layer_id].shape[0] == 62
            assert len(example[2].attributions_mapped[layer_id]) == 62
            assert len(example[2].attributions_bad[layer_id]) == 32
            eval_cls = EvaluateQE()
            score = eval_cls.auc_score(list(zip(*data))[2], 0)
            acc = eval_cls.top1_accuracy(list(zip(*data))[2], 0)
            assert acc == 0.5

            import numpy as np
            eval_cls.attributions_types(data, 0, np.mean)
