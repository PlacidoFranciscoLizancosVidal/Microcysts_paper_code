from torchvision.models import densenet161
from torch import nn
import numpy as np
from PIL import Image
from utilmaps import MapBuilder
import os, sys, cv2, torch
from matplotlib import cm

def addText(image,text, color):
    image = cv2.rectangle(image, (5,5), (80,55), color, -1)
    image = cv2.putText(image,text,
                    (10,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,
                    (0,0,0),
                    3)
    return image

if __name__ == '__main__':
    model = densenet161()
    model.classifier = nn.Linear(model.classifier.in_features, 4)
    model.load_state_dict(torch.load('model.pt',map_location=torch.device('cpu')))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = model.to(device)
    jet_cmap = cm.get_cmap('jet')
    builder = MapBuilder(model,64,60)
    
    image_original =  np.array(Image.open(sys.argv[1]))[:,:,0].astype(float)/255
    image_mask = np.array(Image.open(sys.argv[2]).convert('L')).astype(bool)
    
    generated_map = builder.generateMap(image_original,image_mask)

    coords = np.where(image_mask)[0]
    mn = np.min(coords)
    mx = np.max(coords)

    rgbimage = cv2.cvtColor((image_original*255).astype('uint8'),cv2.COLOR_GRAY2RGB)

    cmemap = cv2.addWeighted((jet_cmap(generated_map[:,:,1])[:,:,0:3]*255).astype('uint8'),0.5,rgbimage,0.5,0)[mn-1:mx+1,:,:]
    drtmap = cv2.addWeighted((jet_cmap(generated_map[:,:,2])[:,:,0:3]*255).astype('uint8'),0.5,rgbimage,0.5,0)[mn-1:mx+1,:,:]
    srdmap = cv2.addWeighted((jet_cmap(generated_map[:,:,3])[:,:,0:3]*255).astype('uint8'),0.5,rgbimage,0.5,0)[mn-1:mx+1,:,:]

    rgbimage = addText(rgbimage,'OCT',(255,255,255))
    cmemap = addText(cmemap,'CME',(0,75,255))
    drtmap = addText(drtmap,'DRT',(255,50,0))
    srdmap = addText(srdmap,'SRD',(75,255,75))

    maps = np.vstack((
            rgbimage,
            cmemap,
            drtmap,
            srdmap
                ))
    im = Image.fromarray(maps)
    im.save(sys.argv[3])
