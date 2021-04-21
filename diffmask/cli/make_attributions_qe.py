import torch
import numpy as np

from diffmask.models.quality_estimation import QualityEstimationRegression
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationBert
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationRoberta
from diffmask.options import make_attributions_parser

from diffmask.attributions.schulz import qe_roberta_schulz_explainer
from diffmask.attributions.guan import qe_roberta_guan_explainer
from diffmask.attributions.integrated_gradient import qe_integrated_gradient_explainer
from diffmask.attributions.attention import qe_roberta_attention_explainer


EXLAINERS = {
    "schulz": qe_roberta_schulz_explainer,
    "guan": qe_roberta_guan_explainer,
    "intergrated_gradient": qe_integrated_gradient_explainer,
    "attention": qe_roberta_attention_explainer,
}


if __name__ == '__main__':
    parser = make_attributions_parser()
    params = parser.parse_args()
    print(params)
    device = "cuda" if params.use_cuda else "cpu"

    if params.seed is not None:
        torch.manual_seed(params.seed)
        np.random.seed(params.seed)

    if params.num_labels > 1:
        if params.architecture == 'roberta':
            qe = QualityEstimationBinaryClassificationRoberta.load_from_checkpoint(params.model_path).to(device)
        elif params.architecture == 'bert':
            qe = QualityEstimationBinaryClassificationBert.load_from_checkpoint(params.model_path).to(device)
        else:
            raise ValueError
    else:
        qe = QualityEstimationRegression.load_from_checkpoint(params.model_path).to(device)

    qe.freeze()
    qe.prepare_data()

    tensor_dataset = qe.test_dataset if params.data_split == 'test' else qe.val_dataset
    text_dataset = qe.test_dataset_orig if params.data_split == 'test' else qe.val_dataset_orig
    attributions = EXLAINERS[params.explainer](
        qe, tensor_dataset, save=params.save, input_only=params.input_only, steps=params.steps,
        batch_size=params.batch_size, num_layers=params.num_layers, learning_rate=params.lr,
        aux_loss_weight=params.aux_loss_weight
    )
