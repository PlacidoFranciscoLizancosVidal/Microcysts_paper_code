from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure

def normalizeImage(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def loadMask(path, size_limit=None):
    with Image.open(path) as im:
        mask = np.array(im.convert('1')).astype(bool)
        
        
        
        if __name__=='__main__':
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(mask)
            
        if size_limit is not None:
            s = generate_binary_structure(2,2)
            label_structures, num_features = label(mask, s)

            for connected_component_id in [x+1 for x in range(num_features)]:
                current_component = label_structures == connected_component_id

                if np.nansum(current_component) > size_limit:
                    mask[current_component] = False
                    
        if __name__=='__main__':
            axs[1].imshow(mask)
            plt.show()
    return mask

def loadImage(path):
    
    with Image.open(path) as im:
        mi_image = normalizeImage(np.array(im.convert('L')))
    return mi_image

if __name__=='__main__':
    loadMask('<path>',2400)