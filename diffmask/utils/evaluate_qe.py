import heapq
import torch

from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot

from diffmask.attributions.schulz import schulz_explainer


def find_word_with_max_attribution(tokens, attributions, words, N=3):
    eos = tokens.index('</s>')
    target_tokens = tokens[eos + 2:]
    target_attributions = attributions[eos + 2:]
    char_to_word = {}
    char_count = 0
    for i, w in enumerate(words):
        for _ in w:
            char_to_word[char_count] = i
            char_count += 1
    char_count = 0
    max_attr_h = []
    word_attr_d = dict()
    curr_word = None
    curr_max = 0
    for tok, attr in zip(target_tokens, target_attributions):
        if tok == '<s>' or tok == '</s>':
            continue
        for _ in tok.lstrip('▁'):
            if curr_word is None:
                curr_word = char_to_word[char_count]
            if curr_word != char_to_word[char_count]:
                heapq.heappush(max_attr_h, (curr_max, curr_word))
                word_attr_d[curr_word] = curr_max * -1
                curr_word = char_to_word[char_count]
                curr_max = 0
            if attr * -1 < curr_max:
                curr_max = attr * -1
            char_count += 1
    heapq.heappush(max_attr_h, (curr_max, curr_word))
    word_attr_d[curr_word] = curr_max
    return [t[1] for t in max_attr_h[:N]], word_attr_d


def get_layer_attributions(model, getter, setter, all_q_z_loc, all_q_z_scale, device, inputs_dict, hidden_state_idx):
    attributions = schulz_explainer(
        model.net,
        inputs_dict={
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["mask"],
            "labels": inputs_dict["labels"],
        },
        getter=getter,
        setter=setter,
        q_z_loc=all_q_z_loc[0].unsqueeze(0).to(device),
        q_z_scale=all_q_z_scale[0].unsqueeze(0).to(device),
        loss_fn=lambda outputs, hidden_states, inputs_dict: outputs[0],
        loss_kl_fn=lambda kl, inputs_dict: (kl * inputs_dict["attention_mask"])
        .mean(-1)
        .mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=10,
        lr=1e-1,
        la=10,
    )
    return attributions


def get_all_attributions(model, getter, setter, all_q_z_loc, all_q_z_scale, device, inputs_dict, num_layers=14, return_mean=False):
    attributions = torch.cat([get_layer_attributions(model, getter, setter, all_q_z_loc, all_q_z_scale, device, inputs_dict, i) for i in range(num_layers)], 0).T
    print(attributions.shape)  # T, L
    if return_mean:
        attributions = torch.mean(attributions, dim=-1)
    attributions = attributions[:inputs_dict["mask"].sum(-1).item()].cpu()
    return attributions


def make_predictions(model, getter, setter, all_q_z_loc, all_q_z_scale, device, input_data, eval_data, num_layers=14, layer_idx=7):
    predictions = []
    max_attributions = []
    for i, item in enumerate(input_data):
        if layer_idx == -1:
            attributions = get_all_attributions(model, getter, setter, all_q_z_loc, all_q_z_scale, device, item, num_layers=num_layers, return_mean=True)
        else:
            attributions = get_layer_attributions(model, getter, setter, all_q_z_loc, all_q_z_scale, device, item, layer_idx)
        attributions = attributions[:item["mask"].sum(-1).item()].cpu()
        error_index, max_word_attributions = find_word_with_max_attribution(
            eval_data[i]['tokens'], attributions.squeeze(), eval_data[i]['words'])
        predictions.append(error_index)
        max_attributions.append(max_word_attributions)
    return predictions, max_attributions


def make_input_data(model, device, use_test=False):
    dataset = model.test_dataset if use_test else model.val_dataset
    dataset_orig = model.test_dataset_orig if use_test else model.val_dataset_orig
    input_data = []
    eval_data = []
    for i in range(len(dataset)):
        input_ids, mask, _, label = [v.to(device) for v in dataset[i]]
        if label.squeeze() == 1:
            inputs_dict = {
                'input_ids': input_ids.unsqueeze(0),
                'mask': mask.unsqueeze(0),
                'labels': label.unsqueeze(0)
            }
            input_data.append(inputs_dict)
            eval_data.append({
                'tokens': model.tokenizer.convert_ids_to_tokens(
                    inputs_dict["input_ids"].squeeze()[:inputs_dict["mask"].sum(-1).item()].squeeze()),
                'words': dataset_orig[i][1].split(),
                'word_labels': dataset_orig[i][3],
                'sent_label': label.unsqueeze(0),
            })
    return input_data, eval_data


def top1_accuracy(eval_data, predictions, attributions):
    # eval_data: List[Dict]
    # predictions: List[List]
    # for each sentence a list of N indexes with highest attribution score
    # attributions: List[Dict]
    # for each sentences a dict with word indices as keys and max attributions score for the given word as values
    total_by_sent = 0
    correct_by_sent = 0
    for i in range(len(eval_data)):
        try:
            assert len(eval_data[i]['word_labels']) == len(attributions[i])
        except AssertionError:
            print('Sequence too long. Skipping')
            continue
        gold_error_indices = set([idx for idx, val in enumerate(eval_data[i]['word_labels']) if val == 1])
        if any([idx in gold_error_indices for idx in predictions[i]]):
            correct_by_sent += 1
        total_by_sent += 1

    print(correct_by_sent)
    print(total_by_sent)
    print('{:.3f}'.format(correct_by_sent / total_by_sent))


def auc_score(eval_data, attributions):
    ys = []
    yhats = []
    for i in range(len(eval_data)):
        try:
            assert len(eval_data[i]['word_labels']) == len(attributions[i])
        except AssertionError:
            print('Sequence too long. Skipping')
            continue
        for idx, val in enumerate(eval_data[i]['word_labels']):
            ys.append(val)
            yhats.append(attributions[i][idx])

    fpr, tpr, _ = roc_curve(ys, yhats)
    score = roc_auc_score(ys, yhats)
    pyplot.plot(fpr, tpr)
    pyplot.show()
    print(score)
