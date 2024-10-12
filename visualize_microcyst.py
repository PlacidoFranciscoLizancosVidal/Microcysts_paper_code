import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, square
from PIL import Image

target_folder = './dataset_original/DATASET_PREPARADO/'
output_folder = './VISUALIZATION/'


set_threshold = 100*100
structuring_component_size = 3


os.makedirs(output_folder,exist_ok=True)

for file_numpy in os.listdir(target_folder):
    current_image = np.load(target_folder + file_numpy)
    
    # Cogemos las imágenes
    ORIGINAL_IMAGE = current_image[:,:,0]
    MASK = current_image[:,:,1]
    DOUBT = current_image[:,:,3]
    CME = current_image[:,:,4]
    
    im = Image.fromarray(CME*255).convert('RGB')
    im.save(output_folder + file_numpy.replace('.npy','') + ".png")
    
    #doubt = closing(DOUBT, square(structuring_component_size))
    #cme = closing(CME, square(structuring_component_size))
    
    #fig, ax = plt.subplots(2)
    #ax[0].imshow(ORIGINAL_IMAGE)
    #ax[1].imshow(CME)
    #ax[2].imshow(cme)
    #plt.show()