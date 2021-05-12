

def update_hparams(ckpt_hparams, new_hparams, overrides=None):
    d_ckpt_hparams = vars(ckpt_hparams)
    d_new_hparams = vars(new_hparams)
    for p, v in d_new_hparams.items():
        if p not in d_ckpt_hparams:
            continue
        if v != d_ckpt_hparams[p]:
            if overrides is not None and p in overrides:
                d_ckpt_hparams[p] = v
            else:
                try:
                    assert d_ckpt_hparams[p] is None
                except AssertionError:
                    print(
                        'Warning! Inconsistent value for parameter {}. '
                        'Value is {} in the checkpoint and {} in the new params'.format(p, d_ckpt_hparams[p], v)
                    )
                    continue
    for p, v in d_ckpt_hparams.items():
        if p not in d_new_hparams:
            print('Value for the argument {} is not provided. Value {} will be inherited from the trained model'.format(
                p, d_ckpt_hparams[p]
            ))
            d_new_hparams[p] = v
    return new_hparams
