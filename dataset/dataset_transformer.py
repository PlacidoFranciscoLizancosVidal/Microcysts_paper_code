import os, shutil
from PIL import Image
import numpy as np
from util_loads import *

window_size = 64
input_folder = './MICROCYSTS/'
mask_folder = './MASKS/'
image_folder = './ORIGINALS/'
target_folder = './SUBMASKS_{}/'.format(window_size)
debug_mode = True # Imprimir imagen original, etiquetas y submascara

try:
    shutil.rmtree(target_folder)
except:
    print('(!)')
os.makedirs(target_folder, exist_ok=False)


def _extractWindow(original_image, coordinate):
    image_f, image_c = original_image.shape
    padded_image = np.pad(original_image[2:-2,2:-2], int(window_size/2+3),mode='reflect')

    coord_f = int(coordinate[0] + window_size/2+1)
    coord_c = int(coordinate[1] + window_size/2+1)

    window = padded_image[coord_f-int(window_size/2):coord_f+int(window_size/2),coord_c-int(window_size/2):coord_c+int(window_size/2)]
    return window

def candidatorFunction(label, mask):
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
    
        base_mask = np.zeros(mask.shape).astype(bool)
        for row in candidate_rows:
            for col in candidate_cols:
                sample_candidate = _extractWindow(label, (row,col))
                
                # Si hay algo de positivo en la ventana, es una ventana candidata que queremos conservar.
                if np.any(sample_candidate):
                    base_mask[max(row - int(window_size/2),startrow) : min(row + int(window_size/2),endrow), max(col - int(window_size/2),startcol) : min(col + int(window_size/2),endcol)] = True


        return base_mask
    
if debug_mode:
    try:
        shutil.rmtree(target_folder[:-1] + '_VIS/')
    except:
        print('[!]')
    os.makedirs(target_folder[:-1] + '_VIS/')
    
for image_name in os.listdir(input_folder):
    if not image_name.startswith('.'):
        loaded_mask = loadMask(mask_folder + image_name)
        loaded_label = loadMask(input_folder + image_name)
        resmask = candidatorFunction(loaded_label,loaded_mask)

        if debug_mode:
            origname = image_name.replace('.png','')
            for imgext in ['png','jpg','jpeg','bmp','tif','tiff']:
                origname = origname.replace(imgext,'.' + imgext)
                
            loaded_image = loadImage(image_folder + origname)*255

            resimg = np.dstack((loaded_label.astype(np.uint8)*255, (resmask^loaded_label).astype(np.uint8)*128, loaded_image))
            im = Image.fromarray(resimg.astype(np.uint8))
            im.save(target_folder[:-1] + '_VIS/' + image_name)
    
        im = Image.fromarray(resmask.astype(np.uint8)*255)
        im.save(target_folder + image_name)