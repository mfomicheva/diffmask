import os
import argparse

from diffmask.models.quality_estimation import QualityEstimationBinaryClassification, QualityEstimationRegression
from diffmask.utils.evaluate_qe import generate_predictions


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
    parser.add_argument("--class_weighting", default=False, action='store_true')
    parser.add_argument("--val_loss", default="f1", choices=["f1", "mcc", "mse"])
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--save_path", default=None, type=str)

    hparams = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpu
    device = "cuda:{}".format(hparams.gpu)

    if hparams.num_labels == 1:
        qe = QualityEstimationRegression
    elif hparams.num_labels == 2:
        qe = QualityEstimationBinaryClassification
    else:
        raise NotImplementedError

    qe = qe.load_from_checkpoint(hparams.model_path).to(device)
    qe.hparams = hparams
    qe.freeze()
    qe.prepare_data()
    loader = qe.test_dataloader()
    predictions = generate_predictions(qe, loader, device, evaluate=True, regression=hparams.num_labels == 1)
    if hparams.save_path is not None:
        with open(hparams.save_path, 'w') as o:
            for p in predictions:
                o.write('{}\n'.format(p))
