import torch

import numpy as np

from diffmask.options import make_attributions_parser

from diffmask.models.quality_estimation import QualityEstimationRegression
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationBert
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationRoberta

from diffmask.cli.util import update_hparams
from diffmask.attributions.util import load_attributions
from diffmask.attributions.attribute_qe import make_data

from diffmask.utils.evaluate_qe import EvaluateQE
from diffmask.utils.metrics import accuracy_precision_recall_f1


if __name__ == '__main__':
    parser = make_attributions_parser()
    params = parser.parse_args()

    device = 'cuda' if params.use_cuda else 'cpu'
    if params.num_labels > 1:
        if params.architecture == 'roberta':
            qe = QualityEstimationBinaryClassificationRoberta.load_from_checkpoint(params.model_path).to(device)
        elif params.architecture == 'bert':
            qe = QualityEstimationBinaryClassificationBert.load_from_checkpoint(params.model_path).to(device)
        else:
            raise ValueError
    else:
        qe = QualityEstimationRegression.load_from_checkpoint(params.model_path).to(device)

    params = update_hparams(qe.hparams, params)
    qe.hparams = params
    print(params)

    qe.freeze()
    qe.prepare_data()

    dataset = qe.test_dataset if params.data_split == 'test' else qe.val_dataset
    orig_dataset = qe.test_dataset_orig if params.data_split == 'test' else qe.val_dataset_orig
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=20)
    if params.num_labels > 1:
        labels = torch.LongTensor([t[2] for t in orig_dataset])
    else:
        labels = torch.FloatTensor([t[2] for t in orig_dataset])
    predictions = EvaluateQE.generate_predictions(qe, data_loader, device=device, evaluate=True,
                                                  regression=params.num_labels == 1)

    print('Total number of samples: {}'.format(len(predictions)))
    print('Histogram predictions')

    if params.num_labels > 1:
        print('Predicted positive class: {}'.format(sum(predictions)))
        random_predictions = torch.randint(0, 2, (len(predictions),))
        accuracy, precision, recall, f1 = accuracy_precision_recall_f1(random_predictions, labels, average=False)
        print((accuracy, precision[1], recall[1], f1[1]))  # do not average, print for class=1
    else:
        labels = labels.detach().cpu().numpy()
        from scipy.stats import pearsonr

        print('Pearson correlation: {}'.format(pearsonr(predictions, labels)[0]))
        print('Mean predictions: {}'.format(np.mean(predictions)))
        print('Median predictions: {}'.format(np.median(predictions)))
        print('Q75 predictions: {}'.format(np.quantile(predictions, 0.75)))
        print('Mean humans: {}'.format(np.mean(labels)))
        print('Median humans: {}'.format(np.median(labels)))
        print('Histogram humans')

    word_labels = []
    for _, _, _, ls in orig_dataset:
        word_labels.extend(ls)

    print('Proportion of positive class labels: {}'.format(sum(word_labels) / len(word_labels)))

    attributions = load_attributions(params.load)
    attributions_data = make_data(dataset, orig_dataset, qe, attributions, silent=False, predictions=predictions)
    attributions_data = [s for s in attributions_data if len(list(set(s[0].word_labels))) != 1]
    if params.num_labels > 1:
        attributions_data = [s for s in attributions_data if s[0].sent_label == 1]
    else:
        if params.threshold is not None:
            if params.threshold == 0.:
                attributions_data = [s for s in attributions_data if 1 in s[0].word_labels]
                # select all that contain at least one positive class label
            else:
                attributions_data = [s for s in attributions_data if s[0].sent_label > params.threshold]
        else:
            attributions_data = [s for s in attributions_data if s[0].sent_label > np.mean(predictions)]
    evaluation = EvaluateQE()
    layer_indexes = list(range(params.num_layers))
    accs = []
    recs = []
    auc_scores = []
    auprc_scores = []
    for lid in layer_indexes:
        score_auc = evaluation.auc_score_per_sample(attributions_data, lid, auprc=False)
        score_auprc = evaluation.auc_score_per_sample(attributions_data, lid, auprc=True)
        acc = evaluation.top1_accuracy(attributions_data, lid)
        rec = evaluation.top1_recall(attributions_data, lid)
        random_acc = evaluation.top1_accuracy(attributions_data, lid, random=True)
        random_rec = evaluation.top1_recall(attributions_data, lid, random=True)
        auc_scores.append(score_auc)
        auprc_scores.append(score_auprc)
        accs.append(acc)
        recs.append(rec)
    print('Best AUC: {}'.format(max(auc_scores)))
    print('Best AUPRC: {}'.format(max(auprc_scores)))
    print('Best ACC@top1: {}'.format(max(accs)))
    print('Best REC@top1: {}'.format(max(recs)))
    print('Random ACC@top1: {}'.format(random_acc))
    print('Random REC@top1: {}'.format(random_rec))
    print('Best layer AUC: {}'.format(np.argmax(auc_scores)))
    print('Best layer AUPRC: {}'.format(np.argmax(auprc_scores)))
    print('Best layer REC@top1: {}'.format(np.argmax(recs)))
    print('Best layer for ACC@top1: {}'.format(np.argmax(accs)))
    print('Total data used for evaluation: {}'.format(len(attributions_data)))

    if params.layer_id is not None:
        score_auc = evaluation.auc_score_per_sample(attributions_data, params.layer_id, auprc=False)
        score_auprc = evaluation.auc_score_per_sample(attributions_data, params.layer_id, auprc=True)
        acc = evaluation.top1_accuracy(attributions_data, params.layer_id)
        rec = evaluation.top1_recall(attributions_data, params.layer_id)
        print('AUC Layer {}: {}'.format(params.layer_id, score_auc))
        print('AUPRC Layer {}: {}'.format(params.layer_id, score_auprc))
        print('ACC@top1 Layer {}: {}'.format(params.layer_id, acc))
        print('REC@top1 Layer {}: {}'.format(params.layer_id, rec))
