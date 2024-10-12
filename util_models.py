# -*- coding: utf-8 -*-
from torchvision.models import *
from torch import nn
import torch
import warnings
    
# Devuelve una red preparada para trabajar con el número de características indicado
def obtainModel(model_name,num_classes, pretrained=False):
    configured_model = None
    configured_model = eval('{}(pretrained={},drop_rate=0.25)'.format(model_name,pretrained))

    if 'efficientnet' in model_name.lower():
        configured_model.classifier = nn.Sequential(
            nn.Dropout(p=configured_model.classifier[0].p, inplace=True),
            nn.Linear(configured_model.classifier[1].in_features, num_classes),
        )
    else:
        configured_model.classifier = nn.Linear(configured_model.classifier.in_features, num_classes)

    return configured_model
