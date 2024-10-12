from torch.utils.data import Dataset
from util_loads import loadMask, loadImage
import os
import numpy as np
from torchvision.transforms import transforms
#from tqdm import tqdm
#import matplotlib.pyplot as plt
#%matplotlib inline
from scipy.ndimage import label, generate_binary_structure

from pathlib import Path
_ = Path

import random
random.seed(512)

from torch import tensor

class MicrocystDataset(Dataset):
    
    def _extractWindow(self,original_image, coordinate):
        image_f, image_c = original_image.shape
        padded_image = np.pad(original_image[2:-2,2:-2], int(self.window_size/2+3),mode='reflect')

        coord_f = int(coordinate[0] + self.window_size/2+1)
        coord_c = int(coordinate[1] + self.window_size/2+1)

        window = padded_image[coord_f-int(self.window_size/2):coord_f+int(self.window_size/2),coord_c-int(self.window_size/2):coord_c+int(self.window_size/2)]
        return window
    
    def _sampleCandidator(self,label, mask):
        window_size = self.window_size
        rows, cols = np.where(mask)

        
        if rows.size > 0:
            startrow = np.min(rows)
            endrow = np.max(rows)
            
            # Aquí debería valer con 0 y tamaño de imagen... pero por si acaso.
            startcol = np.min(cols)
            endcol = np.max(cols)
            
            # El overlap es la mitad de la ventana, pero si la retina no es suficientemente grande, añadimos una fila a mayores por si las moscas.
            candidate_rows = np.arange(startrow, max(startrow+int(window_size/2)+1,endrow), int(window_size/2))
            candidate_cols = np.arange(startcol, endcol, int(window_size/2))
            
            window_centers = []
            for row in candidate_rows:
                for col in candidate_cols:
                    sample_candidate = self._extractWindow(mask, (row,col))
                    alto, ancho = sample_candidate.shape
                    proportional_valid_sample = np.sum(sample_candidate)/(alto*ancho)
        
                    if proportional_valid_sample >= self.valid_surface_threshold:
                        labeled_sample = self._extractWindow(label, (row,col))
                        window_centers.append((row,col,np.any(labeled_sample)))
                    
                        #if np.any(labeled_sample):
                            #plt.imshow(labeled_sample)
                            #plt.show()
            return window_centers             
        else:
            return []
        
    def __init__(self, image_paths, mask_paths, label_paths, window_size, valid_surface_threshold = 0.5, num_cores=8, balance_dataset=False, connected_component_size_limit=None):
        self.valid_surface_threshold = valid_surface_threshold
        self.window_size = window_size
        self.num_cores = num_cores
        self.current_samples = []
        self.current_images = []
        
        iterator = list(zip(image_paths, mask_paths, label_paths))
        for image_path, mask_path, label_path in iterator:
            loaded_image = loadImage(image_path)
            
            current_image_index = len(self.current_images)
            self.current_images.append(loaded_image)
            loaded_mask = loadMask(mask_path)
            loaded_labels = loadMask(label_path,connected_component_size_limit)
            
            self.current_samples += [ (x[0], x[1], x[2], current_image_index) for x in self._sampleCandidator(loaded_labels, loaded_mask)]
        if balance_dataset:
            self.balanceDataset()
            
    def balanceDataset(self):
        num_positives = self.getPositiveLen()
        num_negatives = self.getNegativeLen()

        # Función que coge todos los índices de la clase mayoritaria
        idx = list(np.where([x[2] == (num_positives > num_negatives) for x in self.current_samples])[0])
        
        # Cogemos de la clase mayoritaria el mismo número de índices que la clase minoritaria
        blessed_indexes = random.sample(idx, k = min(num_positives,num_negatives))
        
        # La nueva lista de índices es el TOTAL de la clase minoritaria + los elegidos de la clase mayoritaria
        self.current_samples = [x for x in self.current_samples if x[2] != (num_positives > num_negatives)] + [self.current_samples[idx] for idx in blessed_indexes]
        
        
        
    def __len__(self):
        return len(self.current_samples)
    
    def getPositiveLen(self):
        a = 0
        for x in self.current_samples:
            a+=x[2]
        return a
    
    def getNegativeLen(self):
        a = 0
        for x in self.current_samples:
            a+= (1-x[2])
        return a
    
    def __getitem__(self,idx):
        
        row, col, label, current_image_index = self.current_samples[idx]
        
        sample = self._extractWindow(self.current_images[current_image_index], (row,col))
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64,64)),
            transforms.RandomHorizontalFlip(p=0.5)
            ]
        )
        
        
        target = [1-label, label]
        return t(np.dstack((sample,sample,sample))).float(), tensor(target).float()
        

# Test si el módulo se ejecuta directamente
if __name__ == '__main__':
    oip = _('./dataset/ORIGINALS/')
    loip = list(os.listdir(oip))
    original_images_path =  [_(oip, x) for x in loip if not x.startswith('.')]
    
    tlp = _('./dataset/MICROCYSTS/')
    target_labels_path = [_(tlp, x.replace('.','') + '.png') for x in loip if not x.startswith('.')]
    
    tmp = _('./dataset/MASKS/')
    target_mask_path = [_(tmp, x.replace('.','') + '.png') for x in loip if not x.startswith('.')]
    
    
    test_dataset = MicrocystDataset(original_images_path, target_mask_path, target_labels_path)
    print(len(test_dataset))
    print(test_dataset.getPositiveLen())
    print(test_dataset.getNegativeLen())

    #for image_name in os.listdir(original_images_path):
        
            


    