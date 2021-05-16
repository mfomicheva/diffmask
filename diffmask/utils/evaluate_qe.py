import torch
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import pearsonr
from matplotlib import pyplot

from diffmask.utils.metrics import accuracy_precision_recall_f1, matthews_corr_coef


class EvaluateQE:

    @staticmethod
    def select_data(text_dataset, predictions, params):
        selected = []
        for idx, s in enumerate(text_dataset):
            select = True
            if len(list(set(s[0].word_labels))) == 1:
                select = False
            if params.num_labels > 1 and s[2] != 1:
                select = False
            if params.threshold is not None:
                if params.threshold == 0. and sum(s[0].word_labels) == 0:
                    select = False
                else:
                    if s[2] <= params.threshold:
                        select = False
            else:
                if s[2] <= np.mean(predictions):
                    select = False
            if select:
                selected.append(idx)
        return selected

    @staticmethod
    def select_data_regression(data, max_error=None, max_size=None, min_pred=None, **kwargs):
        data = sorted(data, key=lambda s: s[0].sent_pred, reverse=True)
        if max_error is not None:
            data = [s for s in data if abs(s[0].sent_pred - s[0].sent_label) <= max_error]
        if max_size is not None:
            data = data[:max_size]
        if min_pred is not None:
            data = [s for s in data if s[0].sent_pred > min_pred]
        return data

    @staticmethod
    def select_data_classification(
            data, positive_gold=False, positive_predicted=False, negative_gold=False, negative_predicted=False,
            **kwargs
    ):
        output = []
        for sample in data:
            if positive_gold and sample[0].sent_label != 1:
                continue
            if positive_predicted and sample[0].sent_pred != 1:
                continue
            if negative_gold and sample[0].sent_label == 1:
                continue
            if negative_predicted and sample[0].sent_pred == 1:
                continue
            output.append(sample)
        return output

    @staticmethod
    def attributions_types(data, layer_id, summary_fn):
        all_attributions = []
        src_attributions = []
        tgt_attributions = []
        special_attributions = []
        bad_attributions = []
        for (sample, sample_source_mapped, sample_target_mapped) in data:
            all_attributions.append(summary_fn(sample.select_by_layer(sample.attributions, layer_id).detach().cpu().numpy()))
            src_attributions.append(summary_fn(sample.attributions_source[layer_id].detach().cpu().numpy()))
            tgt_attributions.append(summary_fn(sample.attributions_target[layer_id].detach().cpu().numpy()))
            special_attributions.append(summary_fn(sample.attributions_special[layer_id].detach().cpu().numpy()))
            if len(sample_target_mapped.attributions_bad[layer_id]) > 0:
                bad_attributions.append(summary_fn(sample_target_mapped.attributions_bad[layer_id]))
            else:
                bad_attributions.append(0.)
        res_all = np.mean(all_attributions)
        res_src = np.mean(src_attributions)
        res_tgt = np.mean(tgt_attributions)
        res_spec = np.mean(special_attributions)
        res_bad = np.mean(bad_attributions)
        print('All attributions: {:.4f}'.format(res_all))
        print('Source attributions: {:.4f}'.format(res_src))
        print('Target attributions: {:.4f}'.format(res_tgt))
        print('Special token attributions: {:.4f}'.format(res_spec))
        print('Bad token attributions: {:.4f}'.format(res_bad))
        return res_all, res_src, res_tgt, res_bad, res_spec

    def auc_score_per_sample(self, scores, labels, auprc=False, verbose=False):
        if auprc:
            curve_fn = self._precision_recall_curve
            score_fn = average_precision_score
        else:
            curve_fn = roc_curve
            score_fn = roc_auc_score
        aucs = []
        aucs_rand = []
        assert len(scores) == len(labels)
        for sid in range(len(labels)):
            rand = torch.rand((len(scores[sid]),)).tolist()
            _, _, auc = self._make_curve(labels[sid], scores[sid], curve_fn, score_fn)
            _, _, auc_rand = self._make_curve(labels[sid], rand, curve_fn, score_fn)
            aucs.append(auc)
            aucs_rand.append(auc_rand)
        if verbose:
            print('AUC score: {}'.format(np.mean(aucs)))
            print('AUC score random: {}'.format(np.mean(aucs_rand)))
        return np.mean(aucs)

    def auc_score(self, scores, labels, plot=False, save_plot=None, verbose=False, auprc=False, random_majority=False):
        if auprc:
            curve_fn = self._precision_recall_curve
            score_fn = average_precision_score
            x_label = 'Recall'
            y_label = 'Precision'
        else:
            curve_fn = roc_curve
            score_fn = roc_auc_score
            x_label = 'False Positive Rate'
            y_label = 'True Positive Rate'
        ys, yhats = self._make_flat_data(scores, labels)
        _, yhats_random = self._make_flat_data(scores, labels, random=True, majority=random_majority)
        x, y, score = self._make_curve(ys, yhats, curve_fn, score_fn)
        x_rand, y_rand, score_rand = self._make_curve(ys, yhats_random, curve_fn, score_fn)
        if plot:
            pyplot.plot(x, y)
            pyplot.plot(x_rand, y_rand, linestyle='--', color='black')
            pyplot.xlabel(x_label)
            pyplot.ylabel(y_label)
            if save_plot is not None:
                pyplot.savefig(save_plot)
            else:
                pyplot.show()
        if verbose:
            print('AUC score: {}'.format(score))
            print('AUC score random: {}'.format(score_rand))
        return score

    @staticmethod
    def get_scores_and_labels(data, lid):
        data = list(zip(*data))[2]
        scores = []
        labels = []
        for sample in data:
            labels.append(sample.word_labels)
            scores.append(sample.attributions_mapped[lid])
        return scores, labels

    @staticmethod
    def top1_recall(scores, labels, random=False, verbose=False):
        correct = 0
        assert len(scores) == len(labels)
        for sid in range(len(labels)):
            gold = set([idx for idx, val in enumerate(labels[sid]) if val == 1])
            scores_i = scores[sid]
            if random:
                scores_i = torch.rand((len(scores_i),)).tolist()
            highest_attributions = np.argsort(scores_i)[::-1]
            highest_attributions = highest_attributions[:len(gold)]
            correct_in_sample = len([idx for idx in highest_attributions if idx in gold])
            correct_in_sample = correct_in_sample/len(gold)
            correct += correct_in_sample
        return correct / len(labels)

    @staticmethod
    def top1_accuracy(scores, labels, topk=1, random=False, verbose=False):
        total_by_sent = 0
        correct_by_sent = 0
        assert len(labels) == len(scores)
        for sid in range(len(labels)):
            gold = set([idx for idx, val in enumerate(labels[sid]) if val == 1])
            scores_i = scores[sid]
            if random:
                scores_i = torch.rand((len(scores_i),)).tolist()
            highest_attributions = np.argsort(scores_i)[::-1]
            highest_attributions = highest_attributions[:topk]
            if any([idx in gold for idx in highest_attributions]):
                correct_by_sent += 1
            total_by_sent += 1
        if verbose:
            print('Sentence with correct detection: {}'.format(correct_by_sent))
            print('Total sentences: {}'.format(total_by_sent))
            print('Percentage: {:.3f}'.format(correct_by_sent / total_by_sent))
        return correct_by_sent / total_by_sent

    @staticmethod
    def generate_predictions(model, loader, device, evaluate=False, regression=False):
        all_predictions = []
        all_labels = []
        for batch_idx, sample in enumerate(loader):
            input_ids, mask, _, labels = sample
            inputs_dict = {
                'input_ids': input_ids.to(device),
                'mask': mask.to(device),
                'labels': labels.to(device),
            }
            logits = model(**inputs_dict)[1]
            batch_predictions = logits if regression else logits.argmax(-1)
            all_predictions.append(batch_predictions)
            all_labels.append(labels)

        all_predictions = torch.cat(all_predictions, dim=0).to(device)
        all_labels = torch.cat(all_labels, dim=0).to(device)

        if evaluate:
            if regression:
                print(pearsonr(all_predictions.squeeze().cpu(), all_labels.cpu())[0])
            else:
                accuracy, precision, recall, f1 = accuracy_precision_recall_f1(all_predictions, all_labels, average=False)
                mcc = matthews_corr_coef(all_predictions, all_labels,)
                print((accuracy, precision[1], recall[1], f1[1]))  # do not average, print for class=1
                print(mcc)
                print(sum(all_predictions.tolist()) / len(all_predictions.tolist()))
                print(sum(all_labels.tolist()) / len(all_labels.tolist()))
        return all_predictions.squeeze().tolist()

    @staticmethod
    def _precision_recall_curve(ys, yhats):
        prec, rec, _ = precision_recall_curve(ys, yhats)
        return rec, prec, _

    @staticmethod
    def _make_flat_data(scores, labels, random=False, majority=False):
        ys = []
        yhats = []
        constant = None
        n_sents = len(labels)
        if majority:
            constant = sum([sum(labels[sid]) for sid in range(n_sents)])/sum([len(labels[sid]) for sid in range(n_sents)])
        for sid in range(n_sents):
            scores_i = scores[sid]
            if random:
                if constant is not None:
                    scores_i = torch.full((len(scores[sid]),), constant).tolist()
                else:
                    scores_i = torch.rand((len(scores[sid]),)).tolist()
            for idx, val in enumerate(labels[sid]):
                ys.append(val)
                yhats.append(scores_i[idx])
        return ys, yhats

    @staticmethod
    def _make_curve(ys, yhats, curve_fn, scoring_fn):
        x, y, _ = curve_fn(ys, yhats)
        score = scoring_fn(ys, yhats)
        return x, y, score
