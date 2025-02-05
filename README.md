# Microcysts Source Code

Source Code of the paper: 

```Vidal, P., de Moura, J., Novo, J. et al. Multi-Stage Learning for Intuitive Visualization of Microcystic Macular Edema in OCT Images. J. Med. Biol. Eng. (2025). https://doi.org/10.1007/s40846-025-00930-x```

## 1. Installation Guide

To replicate the experiments, you need the following dependencies and requirements. Ensure you have Python 3.8 installed.

- Clone the repository:
  ```bash
  git clone https://github.com/PlacidoFranciscoLizancosVidal/Microcysts_paper_code.git
  cd Microcysts_paper_code

 # Install the required Python libraries:
 
```bash
pip install -r requirements.txt
```

## 2. Usage Guide

### Train and validation

Python file "ablation_main.py" corresponds to the training and validation of the models.

The rest of files in the root are auxiliar modules:
- dataset_library.py ``` Extracts uniformly distributed windows samples from the images. Implements the Dataset torch class for the loading of samples, returing a balanced dataset.  ```
- util_loads.py ``` Auxiliar image processing functions for dataset_library (such as normalization of the images).  ```
- util_models.py ``` Module to load the models from the Pytorch base. The de-facto hardcoded drop rate is 0.25. ```
- util_storage.py ``` Simple class to load and store the json files. ```
- util_training.py  ``` Implementation of the epoch training and the Early Stopping strategy. ```

For training, validation and testing, the information of each fold must be set in a .json file as desired. In particular, this JSON must contain a list of dictionaries, each dictionary with three components: train, val and test. These correspond to training, validation and testing of each fold. Inside each entry of the dictionary, a list of paths, referencing the image that belongs to said set in that given fold of the crossvalidation. An example of this json file can be found in: "fold_data_example.json"

This project requires three sets of images:

- The original OCT images, without any alteration.
- Binary images, where the microcystic regions are marked in white and the non-relevant regions in black (i.e. binary masks).
- ROI binary images. Represent the regions of the image from which to extract samples.
  - In the first stage, this would be the retinal region of interest.
  - In the second stage, it should be a morphological dilation surrounding the true microcystic patterns.


In ablation_main.py, the configuration of the dataset can be set as follows:
```python
##################################
fold_info = 'FOLD_DATA_FN_L_6V2T1'
p_images = './dataset/ORIGINALS/'
p_labels = './dataset/MICROCYSTS/'
p_roi = './dataset/SUBMASKS_64/'
##################################
```
"fold_info" corresponds to the JSON file with the cross-validation fold information.
"p_images" is the path to the image dataset.
"p_labels" is the path to the binary masks that contain the regions with POSITIVE SAMPLES.
"p_roi" path to the region of the image from where the samples will be extracted.

Secondly, the training configuration can be set in the "c" variable as desired:

```python
c = {
    'experiment_name' : 'Minicist_V5_FASE_2_DENSENET_64x64_ALLREPSFULL',
    'model_name' : 'densenet161',
    'batch_size' : 64,
    'num_workers' : 8,
    'stopping_patience' : 31,
    'scheduler_patience' : 10,
    'scheduler_factor' : 0.75,
    'optimizer_lr' : 0.005,
    'pretrained' : True 
}
```
Finally, "window_ranges" allows to configure a list of window ranges to study in that given stage of the proposed methodology (Stage 1 or 2). For example, to train the first stage, use the full retinal region as binary masks and a window size of 64 by 64. To train the second stage, use the morphologically dilated binary masks of the microcystic regions and a window size of 30 by 30. "ablation_main" will generate automatically a file ending with "_conf" to store the experiment name configuration for reproducibility, as well as folders with the results of training.

### Map generation

In the ```maps``` folder we include the code to generate maps with the given folders.

There are two models included, ```model_30.rar``` and ```model_64.rar```. The 64 model corresponds to the model trained to extract the coarse region of interest. The 30 model, the precise microcysts inside that region.

To generate the maps, execute:

```python generateMicrocystRepresentation.py ./target_image.png ./roi.png ./generated_map.png```

Where ```./target_image.png``` is the original OCT image and ```./roi.png``` represents the retinal region of interest to be analyzed by the models. The map will be saved as the third argument, presented here as ```./generated_map.png```.

## 3. Permanent Links

To enhance reproducibility, the following resources have been archived using Zenodo for persistent access:

https://zenodo.org/records/14030370

## 4. Usage Disclaimer and Citation

This code is provided as-is, intended for statistical purposes only and no private data or fees are required. If you use this code set in your work, please include the following reference:

```P. L. Vidal, J. de Moura, J. Novo, M. Ortega, "Robust fully-automatic multi-stage learning for the intuitive representation of MME accumulations in OCT images", The Visual Computer, 2024 (pending acceptance).```

