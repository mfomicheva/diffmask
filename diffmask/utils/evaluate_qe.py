import torch
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr
from matplotlib import pyplot

from diffmask.attributions.schulz import schulz_explainer, roberta_hidden_states_statistics
from diffmask.utils.metrics import accuracy_precision_recall_f1, matthews_corr_coef
from diffmask.utils.util import map_bpe_moses


class SampleAttributions:

    def __init__(
            self, source_tokens, target_tokens, bpe_tokens, bpe_attributions, word_labels, sent_label,
            sent_pred, layer_id
    ):
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.bpe_tokens = bpe_tokens
        self.bpe_attributions = bpe_attributions
        self.word_labels = word_labels
        self.sent_label = sent_label
        self.sent_pred = sent_pred
        self.layer_id = layer_id

        self.cls_idx = 0
        self.sep_idx = self.bpe_tokens.index('</s>')
        self.eos_idx = len(self.bpe_tokens) - 1

        self.bpe_attributions_layer = None
        self.source_token_attributions = None
        self.target_token_attributions = None
        self.special_token_attributions = None
        self.error_token_attributions = None
        self.random_attributions = None
        self.bpe_moses_target = None

    def map_attributions(self):
        try:
            src_bpe_moses, src_moses_bpe = map_bpe_moses(self.source_bpe_tokens(), self.source_tokens)
            tgt_bpe_moses, tgt_moses_bpe = map_bpe_moses(self.target_bpe_tokens(), self.target_tokens)
        except ValueError:
            raise
        self.set_layer_bpe_attributions()
        self.source_token_attributions = self._max_attribution(src_moses_bpe, self.source_bpe_attributions())
        self.target_token_attributions = self._max_attribution(tgt_moses_bpe, self.target_bpe_attributions())
        self.special_token_attributions = [self.bpe_attributions_layer[idx] for idx in (
            self.cls_idx, self.sep_idx, self.sep_idx + 1, self.eos_idx)]
        self.error_token_attributions = self._error_attributions(
            self.target_bpe_attributions(), self.word_labels, tgt_bpe_moses)

    def set_layer_bpe_attributions(self):
        if self.layer_id == -1:
            self.bpe_attributions_layer = torch.mean(self.bpe_attributions, dim=-1)
        else:
            self.bpe_attributions_layer = self.bpe_attributions[:, self.layer_id]

    def source_bpe_tokens(self):
        return self.bpe_tokens[1:self.sep_idx]

    def target_bpe_tokens(self):
        return self.bpe_tokens[self.sep_idx+2: self.eos_idx]

    def source_bpe_attributions(self):
        return self.bpe_attributions_layer[1:self.sep_idx]

    def target_bpe_attributions(self):
        return self.bpe_attributions_layer[self.sep_idx+2:self.eos_idx]

    @staticmethod
    def _error_attributions(attributions, labels, mapping):
        error_idxs = [i for i, l in enumerate(labels) if l == 1]
        return [a for i, a in enumerate(attributions) if mapping[i] in error_idxs]

    @staticmethod
    def _max_attribution(moses_to_bpe, bpe_attributions):
        token_indexes = sorted(moses_to_bpe.keys())
        return [max([bpe_attributions[bpe_idx] for bpe_idx in moses_to_bpe[index]]) for index in token_indexes]


class EvaluateQE:

    def __init__(self, model, getter, setter, layer_indexes, device, split='valid'):
        self.model = model
        self.getter = getter
        self.setter = setter
        self.layer_indexes = layer_indexes
        self.split = split
        self.device = device

        self.loader = self.model.test_dataloader() if split == 'test' else self.model.val_dataloader()
        self.text_dataset = self.model.test_dataset_orig if split == 'test' else self.model.val_dataset_orig
        self.dataset = self.model.test_dataset if split == 'test' else self.model.val_dataset
        self.attributions = None

    def attribution_schulz(self):
        all_q_z_loc, all_q_z_scale = roberta_hidden_states_statistics(self.model)
        result = []
        for batch_idx, sample in enumerate(self.loader):
            input_ids, mask, _, labels = sample
            inputs_dict = {
                'input_ids': input_ids.to(self.device),
                'attention_mask': mask.to(self.device),
                'labels': labels.to(self.device),
            }
            all_attributions = []
            for layer_idx in self.layer_indexes:
                layer_attributions = schulz_explainer(
                    self.model.net,
                    inputs_dict=inputs_dict,
                    getter=self.getter,
                    setter=self.setter,
                    q_z_loc=all_q_z_loc[0].unsqueeze(0).to(self.device),
                    q_z_scale=all_q_z_scale[0].unsqueeze(0).to(self.device),
                    loss_fn=lambda outputs, hidden_states, inputs_dict: outputs[0],
                    loss_kl_fn=lambda kl, inputs_dict: (kl * inputs_dict["attention_mask"])
                        .mean(-1)
                        .mean(-1),
                    hidden_state_idx=layer_idx,
                    steps=10,
                    lr=1e-1,
                    la=10,
                )
                all_attributions.append(layer_attributions.unsqueeze(-1))
            all_attributions = torch.cat(all_attributions, -1)  # B, T, L
            for bidx in range(all_attributions.shape[0]):
                try:
                    result.append(all_attributions[bidx, :, :])  # T, L
                except IndexError:
                    break
        self.attributions = result

    def select_target_data(self, layer_id, ignore_correct_gold=True, ignore_correct_predicted=True, predictions=None):
        if layer_id != -1:
            assert layer_id in self.layer_indexes
            layer_id = self.layer_indexes.index(layer_id)
        res = []
        for sentid in range(len(self.attributions)):
            input_ids, mask, _, sent_labels = self.dataset[sentid]
            bpe_tokens = self.model.tokenizer.convert_ids_to_tokens(input_ids.squeeze()[:mask.sum(-1).item()].squeeze())
            sent_pred = predictions[sentid] if predictions is not None else None
            bpe_attributions = self.attributions[sentid][:mask.sum(-1).item()].cpu()
            if ignore_correct_gold and sent_labels.item() != 1:
                continue
            if ignore_correct_predicted and sent_pred is not None and sent_pred != 1:
                continue
            sample = SampleAttributions(
                self.text_dataset[sentid][0].split(), self.text_dataset[sentid][1].split(), bpe_tokens,
                bpe_attributions, self.text_dataset[sentid][3], sent_labels.item(), sent_pred, layer_id
            )
            try:
                sample.map_attributions()
            except ValueError:
                print('BPE mapping error! Skipping...')
                continue
            try:
                assert len(sample.word_labels) == len(sample.target_token_attributions)
            except AssertionError:
                print('Sequence too long. Skipping')
                continue
            if len(sample.source_tokens) == 0 or len(sample.target_tokens) == 0 or len(sample.bpe_tokens) == 0:
                print('Empty segment! Skipping...')
                continue
            res.append(sample)
        return res

    def generate_predictions(self, evaluate=False, regression=False):
        return generate_predictions(self.model, self.loader, self.device, evaluate=evaluate, regression=regression)

    @staticmethod
    def attributions_types(data):
        all_attributions = []
        src_attributions = []
        tgt_attributions = []
        special_attributions = []
        bad_attributions = []
        for sample in data:
            all_attributions.extend(sample.bpe_attributions_layer)
            src_attributions.extend(sample.source_bpe_attributions())
            tgt_attributions.extend(sample.target_bpe_attributions())
            special_attributions.extend(sample.special_token_attributions)
            bad_attributions.extend(sample.error_token_attributions)
        print('Mean all attributions: {}'.format(np.mean(all_attributions)))
        print('Mean source attributions: {}'.format(np.mean(src_attributions)))
        print('Mean target attributions: {}'.format(np.mean(tgt_attributions)))
        print('Mean special token attributions: {}'.format(np.mean(special_attributions)))
        print('Mean bad token attributions: {}'.format(np.mean(bad_attributions)))

    @staticmethod
    def auc_score(data, random=False):
        ys = []
        yhats = []
        for i, sample in enumerate(data):
            if random:
                attributions = torch.rand((len(sample.target_token_attributions),)).tolist()
            else:
                attributions = sample.target_token_attributions
            for idx, val in enumerate(sample.word_labels):
                ys.append(val)
                yhats.append(attributions[idx])
        fpr, tpr, _ = roc_curve(ys, yhats)
        score = roc_auc_score(ys, yhats)
        pyplot.plot(fpr, tpr)
        pyplot.show()
        print(score)

    @staticmethod
    def top1_accuracy(data, topk=1, random=False, ):
        # data: output of self.select_target()
        total_by_sent = 0
        correct_by_sent = 0
        for i, sample in enumerate(data):
            gold = set([idx for idx, val in enumerate(sample.word_labels) if val == 1])
            if random:
                attributions = torch.rand((len(sample.target_token_attributions),)).tolist()
            else:
                attributions = sample.target_token_attributions
            highest_attributions = np.argsort(attributions)[::-1]
            highest_attributions = highest_attributions[:topk]
            if any([idx in gold for idx in highest_attributions]):
                correct_by_sent += 1
            total_by_sent += 1
        print(correct_by_sent)
        print(total_by_sent)
        print('{:.3f}'.format(correct_by_sent / total_by_sent))


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
