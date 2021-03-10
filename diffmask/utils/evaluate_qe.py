import torch
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot

from diffmask.attributions.schulz import schulz_explainer, roberta_hidden_states_statistics


class AttributionsQE:

    def __init__(self, model, getter, setter, layer_indexes, device, use_mean=False, split='valid'):
        self.model = model
        self.getter = getter
        self.setter = setter
        self.layer_indexes = layer_indexes
        self.use_mean = use_mean
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

    def select_target_data(self, layer_id):
        if layer_id != -1:
            assert layer_id in self.layer_indexes
            layer_id = self.layer_indexes.index(layer_id)
        res = []
        for sentid in range(len(self.attributions)):
            src = self.text_dataset[sentid][0].split()
            target_words = self.text_dataset[sentid][1].split()
            input_ids, mask, _, labels = self.dataset[sentid]
            tokens = self.model.tokenizer.convert_ids_to_tokens(input_ids.squeeze()[:mask.sum(-1).item()].squeeze())
            token_attributions = self.attributions[sentid][:mask.sum(-1).item()].cpu()
            tgt_start_idx = tokens.index('</s>') + 2
            target_tokens = tokens[tgt_start_idx:]
            target_token_attributions_all = token_attributions[tgt_start_idx:, :]
            if layer_id == -1:
                target_token_attributions = torch.mean(target_token_attributions_all, dim=-1)
            else:
                target_token_attributions = target_token_attributions_all[:, layer_id]
            target_word_attributions = self._find_word_with_max_attribution(target_tokens, target_token_attributions, target_words)
            item = {
                'source': src,
                'target_words': target_words,
                'target_tokens': target_tokens,
                'target_word_attributions': target_word_attributions,
                'target_token_attributions_all_layers': target_token_attributions_all,
                'target_token_attributions': target_token_attributions.squeeze(),
                'word_labels': self.text_dataset[sentid][3],
                'sent_label': labels.item(),
            }
            res.append(item)
        return res

    @staticmethod
    def top1_accuracy(data, ignore=True, topk=1, silent=False):
        # data: output of self.select_target()
        total_by_sent = 0
        correct_by_sent = 0
        for item in data:
            if ignore and item['sent_label'] != 1:
                continue
            try:
                assert len(item['word_labels']) == len(item['target_word_attributions'])
            except AssertionError:
                if not silent:
                    print('Sequence too long. Skipping')
                continue
            gold = set([idx for idx, val in enumerate(item['word_labels']) if val == 1])
            predictions = np.argsort(item['target_word_attributions'])[::-1]
            predictions = predictions[:topk]
            if any([idx in gold for idx in predictions]):
                correct_by_sent += 1
            total_by_sent += 1
        print(correct_by_sent)
        print(total_by_sent)
        print('{:.3f}'.format(correct_by_sent / total_by_sent))

    @staticmethod
    def auc_score(data, ignore=True, silent=False):
        ys = []
        yhats = []
        for item in data:
            if ignore and item['sent_label'] != 1:
                continue
            try:
                assert len(item['word_labels']) == len(item['target_word_attributions'])
            except AssertionError:
                if not silent:
                    print('Sequence too long. Skipping')
                continue
            for idx, val in enumerate(item['word_labels']):
                ys.append(val)
                yhats.append(item['target_word_attributions'][idx])
        fpr, tpr, _ = roc_curve(ys, yhats)
        score = roc_auc_score(ys, yhats)
        pyplot.plot(fpr, tpr)
        pyplot.show()
        print(score)

    @staticmethod
    def _find_word_with_max_attribution(target_tokens, target_attributions, words):
        char_to_word = {}
        char_count = 0
        for i, w in enumerate(words):
            for _ in w:
                char_to_word[char_count] = i
                char_count += 1
        char_count = 0
        word_attr = dict()
        curr_word = None
        curr_max = 0
        assert len(target_attributions) == len(target_tokens)
        for tok, attr in zip(target_tokens, target_attributions):
            if tok == '<s>' or tok == '</s>':
                continue
            for _ in tok.lstrip('▁'):
                if curr_word is None:
                    curr_word = char_to_word[char_count]
                if curr_word != char_to_word[char_count]:
                    word_attr[curr_word] = curr_max * -1
                    curr_word = char_to_word[char_count]
                    curr_max = 0
                if attr * -1 < curr_max:
                    curr_max = attr * -1
                char_count += 1
        word_attr[curr_word] = curr_max * -1
        return [word_attr[idx].item() for idx in range(len(word_attr))]
