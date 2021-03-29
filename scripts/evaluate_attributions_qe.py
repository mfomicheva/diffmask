import os
import argparse
import pytorch_lightning as pl

from diffmask.models.quality_estimation import QualityEstimationBinaryClassification, QualityEstimationRegression
from diffmask.utils.evaluate_qe import EvaluateQE
from diffmask.attributions.schulz import roberta_hidden_states_statistics, schulz_explainer
from diffmask.utils.getter_setter import roberta_getter, roberta_setter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model", type=str, default="xlm-roberta-base")
    parser.add_argument("--src_train_filename", type=str)
    parser.add_argument("--tgt_train_filename", type=str)
    parser.add_argument("--labels_train_filename", type=str)
    parser.add_argument("--word_labels_train_filename", type=str, default=None)
    parser.add_argument("--src_val_filename", type=str)
    parser.add_argument("--tgt_val_filename", type=str)
    parser.add_argument("--labels_val_filename", type=str)
    parser.add_argument("--word_labels_val_filename", type=str, default=None)
    parser.add_argument("--src_test_filename", type=str)
    parser.add_argument("--tgt_test_filename", type=str)
    parser.add_argument("--labels_test_filename", type=str)
    parser.add_argument("--word_labels_test_filename", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=float, default=0)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--class_weighting", default=False, action='store_true')
    parser.add_argument("--val_loss", default="f1", choices=["f1", "mcc", "mse"])
    parser.add_argument("--num_labels", default=1, type=int)
    parser.add_argument("--num_layers", default=12, type=int)
    parser.add_argument("--save", default=None, type=str)
    parser.add_argument("--load", default=None, type=str)
    parser.add_argument("--regression_threshold", default=None, type=float)

    hparams = parser.parse_args()
    print(hparams)
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpu

    device = "cuda:{}".format(hparams.gpu)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=hparams.model_path,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
    )

    assert os.path.exists(hparams.model_path)
    if hparams.num_labels > 1:
        qe = QualityEstimationBinaryClassification.load_from_checkpoint(hparams.model_path).to(device)
    else:
        qe = QualityEstimationRegression.load_from_checkpoint(hparams.model_path).to(device)

    qe.hparams = hparams  # in case test path changed
    print('Loaded existing model...')

    qe.freeze()
    qe.prepare_data()

    layer_indexes = list(range(hparams.num_layers))
    attributions_split = 'test' if hparams.src_test_filename is not None else 'valid'
    attributions_qe = EvaluateQE(qe, roberta_getter, roberta_setter, layer_indexes, device, split=attributions_split)

    predictions = attributions_qe.generate_predictions(evaluate=True, regression=hparams.num_labels == 1)

    attributions_qe.attribution_schulz(save=hparams.save, load=hparams.load)

    for layerid in range(hparams.num_layers):
        data = attributions_qe.select_target_data(
            layerid, ignore_correct_gold=True, ignore_correct_predicted=True, predictions=predictions,
            regression_threshold=hparams.regression_threshold
        )

        print('Random')
        attributions_qe.top1_accuracy(data, random=True)
        attributions_qe.auc_score(data, random=True)

        print('Test layer {}'.format(layerid))
        attributions_qe.top1_accuracy(data)
        attributions_qe.auc_score(data)
        attributions_qe.attributions_types(data)