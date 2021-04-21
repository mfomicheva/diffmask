import pickle
import torch
from tqdm.auto import tqdm
from ..utils.getter_setter import (
    label_getter,
    roberta_getter,
    roberta_setter,
    bert_getter,
    bert_setter,
    gru_getter,
    gru_setter,
)


def integrated_gradient(
    model, inputs_dict, getter, setter, label_getter, hidden_state_idx=0, steps=10
):

    with torch.no_grad():
        _, hidden_states = getter(model, inputs_dict)

    hidden_states[hidden_state_idx].requires_grad_(True)

    grads = (
        sum(
            torch.autograd.grad(
                label_getter(
                    setter(
                        model,
                        inputs_dict,
                        hidden_states=[None] * hidden_state_idx
                        + [hidden_states[hidden_state_idx] * alpha]
                        + [None] * (len(hidden_states) - hidden_state_idx - 1),
                    )[0],
                    inputs_dict,
                ).sum(-1),
                hidden_states[hidden_state_idx],
            )[0]
            for alpha in tqdm(torch.linspace(0, 1, steps))
        )
        / steps
    )

    attributions = hidden_states[hidden_state_idx].detach() * grads

    return attributions


def qe_integrated_gradient_explainer(
        qe_model, tensor_dataset, save=None, load=None, steps=50, batch_size=1, num_layers=14, learning_rate=1e-1,
        aux_loss_weight=10, verbose=False, num_workers=20
):

    if load is not None:
        result = pickle.load(open(load, 'rb'))
        return result

    device = next(qe_model.parameters()).device
    result = []
    loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, num_workers=num_workers)
    for batch_idx, sample in enumerate(loader):
        input_ids, mask, _, labels = sample
        inputs_dict = {
            'input_ids': input_ids.to(device),
            'attention_mask': mask.to(device),
            'labels': labels.to(device),
        }
        all_attributions = []
        for layer_idx in range(num_layers):
            layer_attributions = integrated_gradient(
                model=qe_model.net,
                inputs_dict=inputs_dict,
                getter=roberta_getter,
                setter=roberta_setter,
                label_getter=label_getter,
                hidden_state_idx=layer_idx,
                steps=steps,
            ).sum(-1).abs()
            all_attributions.append(layer_attributions.unsqueeze(-1))
        all_attributions = torch.cat(all_attributions, -1)  # B, T, L
        for bidx in range(all_attributions.shape[0]):
            try:
                result.append(all_attributions[bidx, :, :])  # T, L
            except IndexError:
                break
    if save is not None:
        pickle.dump(result, open(save, 'wb'))
    return result


def sst_bert_integrated_gradient(model, inputs_dict, hidden_state_idx=0, steps=10):
    return integrated_gradient(
        model=model.net,
        inputs_dict={
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["mask"],
            "labels": inputs_dict["labels"],
        },
        getter=bert_getter,
        setter=bert_setter,
        label_getter=label_getter,
        hidden_state_idx=hidden_state_idx,
        steps=steps,
    )


def sst_gru_integrated_gradient(model, inputs_dict, hidden_state_idx=0, steps=10):
    model.train()
    output = integrated_gradient(
        model=model,
        inputs_dict=inputs_dict,
        getter=gru_getter,
        setter=gru_setter,
        label_getter=lambda outputs, inputs_dict: outputs[0][
            range(len(outputs[0])), inputs_dict["labels"]
        ],
        hidden_state_idx=hidden_state_idx,
        steps=steps,
    )
    model.eval()
    return output


def squad_bert_integrated_gradient(model, inputs_dict, hidden_state_idx=0, steps=10):
    return integrated_gradient(
        model=model.net,
        inputs_dict={
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["mask"],
            "token_type_ids": inputs_dict["token_type_ids"],
            "start_positions": inputs_dict["start_positions"],
            "end_positions": inputs_dict["end_positions"],
        },
        getter=bert_getter,
        setter=bert_setter,
        label_getter=lambda outputs, inputs_dict: outputs[1][
            range(len(outputs[1])), inputs_dict["start_positions"]
        ]
        + outputs[2][range(len(outputs[2])), inputs_dict["end_positions"]],
        hidden_state_idx=hidden_state_idx,
        steps=steps,
    )
