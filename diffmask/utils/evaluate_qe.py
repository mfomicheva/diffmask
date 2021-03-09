from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot


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
    for i in range(len(eval_data['word_labels'])):
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
