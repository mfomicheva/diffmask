from diffmask.models.quality_estimation import QualityEstimationRegression
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationBert
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationRoberta
from diffmask.utils.getter_setter import roberta_getter, roberta_setter
from diffmask.attributions.attribute_qe import AttributeQE
from diffmask.options import make_parser


if __name__ == '__main__':
    parser = make_parser()
    parser.add_argument("--num_layers", default=14, type=int)
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--input_only", default=False, action="store_true")
    parser.add_argument("--save", default=None, type=str)
    parser.add_argument("--data_split", default="valid", choices=["test", "valid"])
    parser.add_argument("--batch_size", default=None, type=int)
    hparams = parser.parse_args()
    print(hparams)
    device = "cuda" if hparams.use_cuda else "cpu"

    if hparams.num_labels > 1:
        if hparams.architecture == 'roberta':
            qe = QualityEstimationBinaryClassificationRoberta.load_from_checkpoint(hparams.model_path).to(device)
        elif hparams.architecture == 'bert':
            qe = QualityEstimationBinaryClassificationBert.load_from_checkpoint(hparams.model_path).to(device)
        else:
            raise ValueError
    else:
        qe = QualityEstimationRegression.load_from_checkpoint(hparams.model_path).to(device)
    qe.hparams = hparams  # in case test path changed

    qe.freeze()
    qe.prepare_data()

    layer_indexes = list(range(hparams.num_layers))
    split = 'test' if hparams.src_test_filename is not None else 'valid'
    attribute_qe = AttributeQE(
        qe, roberta_getter, roberta_setter, layer_indexes, device, split=split, batch_size=hparams.batch_size)
    attribute_qe.make_attributions(save=hparams.save, input_only=hparams.input_only, steps=hparams.steps)
