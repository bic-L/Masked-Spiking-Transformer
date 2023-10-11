# --------------------------------------------------------
# Masked Spiking Transformer
# --------------------------------------------------------

from .mst import MaskedSpikingTransformer


def build_model(config, args):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'mst':
        model = MaskedSpikingTransformer(img_size=config.DATA.IMG_SIZE,
                                         patch_size=config.MODEL.MST.PATCH_SIZE,
                                         in_chans=config.MODEL.MST.IN_CHANS,
                                         num_classes=config.MODEL.NUM_CLASSES,
                                         embed_dim=config.MODEL.MST.EMBED_DIM,
                                         depths=config.MODEL.MST.DEPTHS,
                                         num_heads=config.MODEL.MST.NUM_HEADS,
                                         window_size=config.MODEL.MST.WINDOW_SIZE,
                                         mlp_ratio=config.MODEL.MST.MLP_RATIO,
                                         qkv_bias=config.MODEL.MST.QKV_BIAS,
                                         qk_scale=config.MODEL.MST.QK_SCALE,
                                         drop_rate=config.MODEL.DROP_RATE,
                                         drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                         ape=config.MODEL.MST.APE,
                                         norm_layer=layernorm,
                                         patch_norm=config.MODEL.MST.PATCH_NORM,
                                         use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                         fused_window_process=config.FUSED_WINDOW_PROCESS,
                                         masking_ratio=args.masking_ratio,
                                         dataset=args.dataset,)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
