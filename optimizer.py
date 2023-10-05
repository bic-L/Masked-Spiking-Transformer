# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim

try:
    from apex.optimizers import FusedAdam, FusedLAMB
except:
    FusedAdam = None
    FusedLAMB = None
    print("To use FusedLAMB or FusedAdam, please install apex.")
    
    
def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False
    
def regular_set(model, paras=([],[],[],[])):
    for n, module in model._modules.items():
        if isActivation(module.__class__.__name__.lower()) and hasattr(module, "up"):
            for name, para in module.named_parameters():
                if not para.requires_grad:
                    continue  # frozen weights
                if name.endswith(".bias"):
                    paras[3].append(para)
                    print(f"{name} 3")
                else:
                    paras[0].append(para)
                    print(f"{name} 0")
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                if not para.requires_grad:
                    continue  # frozen weights
                if name.endswith(".bias"):
                    paras[3].append(para)
                    print(f"{name} 3")
                else:
                    paras[2].append(para)
                    print(f"{name} 2")
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                if not para.requires_grad:
                    continue  # frozen weights
                if name.endswith(".bias"):
                    paras[3].append(para)
                    print(f"{name} 3")
                else:
                    paras[1].append(para)
                    print(f"{name} 1")
    return paras


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)
    
    # para1, para2, para3, para4 = regular_set(model)
    

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
        # optimizer = optim.AdamW([
        #                         {'params': para1, 'weight_decay': 5e-4, "lr" : 1e-5 }, 
        #                         {'params': para2, 'weight_decay': config.TRAIN.WEIGHT_DECAY}, 
        #                         {'params': para3, 'weight_decay': config.TRAIN.WEIGHT_DECAY},
        #                         {'params': para4, 'weight_decay': 0.},
        #                         ], eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
        #                         lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'fused_adam':
        optimizer = FusedAdam(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'fused_lamb':
        optimizer = FusedLAMB(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    up = []
    

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        # if (len(param.shape) == 1 and 'up' not in name) or name.endswith(".bias") or (name in skip_list) or \
        #         check_keywords_in_name(name, skip_keywords):
        if (len(param.shape) == 1) or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        elif ('up' in name):
            up.append(param)  
            # print(f"{name} threshold")               
        else:
            has_decay.append(param)
    # return [{'params': has_decay},
    #         {'params': up, 'weight_decay': 5e-4},
    #         {'params': no_decay, 'weight_decay': 0.}]
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
