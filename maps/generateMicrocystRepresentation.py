from torchvision.models import densenet161
from torch import nn
import numpy as np
from PIL import Image
from utilmaps import *
import os, sys, cv2, torch
from matplotlib import cm
import scipy as sp
import scipy.ndimage

  if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    jet_cmap = cm.get_cmap('jet')

    image_original =  np.array(Image.open(sys.argv[1]))[:,:,0].astype(float)/255
    image_mask = np.array(Image.open(sys.argv[2]).convert('L')).astype(bool)    
    
    # STAGE 1
    model = densenet161()
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load('model_64.pt',map_location=torch.device('cpu')))
    model = model.to(device)
    builder = MapBuilder(model,64,32,model_input_size=224)

    roi_mask = builder.generateMap(image_original,image_mask)*image_mask
    roi_mask = roi_mask > 0.1
    
    del model
    torch.cuda.empty_cache()
    
    # STAGE 2
    model = densenet161()
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load('model_30.pt',map_location=torch.device('cpu')))
    model = model.to(device)
    
    builder = MapBuilder(model,30,25,model_input_size=64)
    microcyst_map = np.nan_to_num(builder.generateMap(image_original,roi_mask))
    
    microcyst_map = sp.ndimage.filters.gaussian_filter(microcyst_map, 3, mode='constant')
    
    rgbimage = cv2.cvtColor((image_original*255).astype('uint8'),cv2.COLOR_GRAY2RGB)
    generated_map = cv2.addWeighted((jet_cmap(microcyst_map)[:,:,0:3]*255).astype('uint8'),0.5,rgbimage,0.5,0)  * np.dstack((image_mask,image_mask,image_mask))

        
    im = Image.fromarray(generated_map)
    im.save(sys.argv[3])
