#### IMPORTS ####
import os
from pathlib import Path
_ = Path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#### PATHS ####
path_labels = _("./LABELS/")
path_masks = _("./MASKS/")
path_originals = _("./ORIGINALS/")
path_dest = _("./DATASET_PREPARADO/")
###############
os.makedirs(path_dest,exist_ok=True)

def maskToLabels(input_image, definition ):
    definition = definition.upper()
    R = np.array(input_image[:,:,0], dtype=bool)
    G = np.array(input_image[:,:,1], dtype=bool)
    B = np.array(input_image[:,:,2], dtype=bool)
    
    return eval(definition)

#### CARGA DE IMÁGENES ####
for image_name in os.listdir(path_originals):
    
    if 'ipynb_checkpoints' in image_name:
        continue
    
    labels = np.array(Image.open(_(path_labels,image_name.replace('.','')+'.png')).convert('RGB'))
    mask = np.array(Image.open(_(path_masks,image_name.replace('.','')+'.png')).convert('RGB'))
    original_image = np.array(Image.open(_(path_originals,image_name)).convert('L')).astype(float)
    ORIGINAL_IMAGE = ((original_image - np.min(original_image))/np.ptp(original_image))

    CME = maskToLabels(labels, "~R & ~G & B") # Azul
    DRT = maskToLabels(labels, "R & ~G & ~B") # Rojo
    SRD = maskToLabels(labels, "~R & G & ~B") # Verde
    HEALTHY = maskToLabels(labels, "R & G & B") # Blanco
    DOUBT = maskToLabels(mask, "R & G & B") & maskToLabels(labels, "~(R | G | B)") # Doubt
    MASK = maskToLabels(mask, "R & G & B") # Mask

    np.save(_(path_dest,image_name.replace('.','')+'.npy'),np.dstack((ORIGINAL_IMAGE, MASK, HEALTHY, DOUBT, CME, DRT, SRD)))