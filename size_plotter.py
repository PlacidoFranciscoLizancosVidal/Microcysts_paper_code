import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.ndimage import label, generate_binary_structure
import numpy as np
import os
from PIL import Image
target_folder = './dataset/MICROCYSTS/'

def assignLabels(image_path):
    with Image.open(image_path) as img:
        img = img.convert('1')
        arr_img = np.array(img)
        mask = arr_img > 0
        
        results = []
        s = generate_binary_structure(2,2)
        label_structures, num_features = label(mask, s)

        for connected_component_id in [x+1 for x in range(num_features)]:
            current_component = label_structures == connected_component_id
            current_component_size = np.nansum(current_component)

            results.append(current_component_size)
    return results


if __name__ == '__main__':

    p = Pool(8)
    lm = p.map(assignLabels, [target_folder + x for x in os.listdir(target_folder) if not x.startswith('.')])
    p.close()
    
    result_list = []
    for x in lm:
        result_list+=x
        
    plt.hist(result_list,bins=1000,color='k')
    plt.axvline(np.median(result_list),color='g')
    plt.axvline(np.median(result_list) + np.std(result_list)*1.25,color='r')
    plt.axvline(np.median(result_list) - np.std(result_list)*1.25,color='r')
    
    print(np.median(result_list) + np.std(result_list)*1.25)
    print(np.median(result_list))
    ax = plt.gca()
    ax.set_xlim(0, 5000)
    #ax.set_ylim(0, 384)


    plt.show()

