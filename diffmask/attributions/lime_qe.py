import pickle
import torch
import numpy as np

from tqdm import tqdm

from diffmask.models.quality_estimation import make_tensors
from diffmask.models.quality_estimation import QualityEstimationRegression

from lime.lime_text import LimeTextExplainer


def predict(qe_model, srcs, tgts, labels):
    tensor_dataset, _ = make_tensors(srcs, tgts, labels, qe_model.tokenizer, 'roberta')
    loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=1, shuffle=False, num_workers=20)
    device = next(qe_model.parameters()).device
    res = []
    for sample in loader:
        input_ids, mask, _, labels = sample
        inputs_dict = {
            'input_ids': input_ids.to(device),
            'mask': mask.to(device),
            'labels': labels.to(device),
        }
        logits = qe_model(**inputs_dict)[1].cpu()
        res.append(logits)
    return res


def explain_instance(explainer, qe_model, text_a, text_b, mode='regression'):
    def predict_fn(texts):
        res = predict(qe_model, [text_a] * len(texts), texts, torch.zeros((len(texts),)))
        if mode == 'regression':
            res = np.vstack((res, res)).T
        else:
            res = np.vstack(res)
        return res
    preds = predict(qe_model, [text_a], [text_b], torch.zeros((1,)))
    exp = explainer.explain_instance(text_b, predict_fn, num_features=len(text_b.split()), labels=(1,))
    return preds, exp.as_map()


def qe_lime_explainer(
        qe_model, tensor_dataset, text_dataset, save=None, load=None, steps=50, batch_size=1, num_layers=14,
        learning_rate=1e-1, aux_loss_weight=10, verbose=False, num_workers=20, input_only=False,
):
    explainer = LimeTextExplainer(class_names=['score', 'score'], bow=False, split_expression=' ')
    mode = 'regression' if type(qe_model) is QualityEstimationRegression else 'classification'
    result = []
    for idx in tqdm(range(len(text_dataset))):
        src = text_dataset[idx][0]
        tgt = text_dataset[idx][1]
        pred, exp = explain_instance(explainer, qe_model, src, tgt, mode=mode)
        exp = exp[1]
        feature_maps = np.zeros(len(exp))
        for k, v in exp:
            feature_maps[k] = v
        attributions = torch.as_tensor(feature_maps).unsqueeze(1)
        result.append(attributions)
    if save is not None:
        pickle.dump(result, open(save, 'wb'))
    return result
