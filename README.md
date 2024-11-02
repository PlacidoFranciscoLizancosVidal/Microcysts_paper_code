# Microcysts Source Code

Source Code of the paper: 

P. L. Vidal, J. de Moura, J. Novo, M. Ortega, "Robust fully-automatic multi-stage learning for the intuitive representation of MME accumulations in OCT images", The Visual Computer, 2024 (pending acceptance).

## 1. Installation Guide

To replicate the experiments, you need the following dependencies and requirements. Ensure you have Python 3.8 installed.

- Clone the repository:
  ```bash
  git clone https://github.com/PlacidoFranciscoLizancosVidal/Microcysts_paper_code.git
  cd Microcysts_paper_code

 # Install the required Python libraries:
 
pip install -r requirements.txt

## 2. Usage Guide

### Train and validation

Python file "ablation_main.py" corresponds to the training and validation of the models.

For training, validation and testing, the information of each fold must be set in a .json file as desired. In particular, this JSON must contain a list of dictionaries, each dictionary with three components: train, val and test. These correspond to training, validation and testing of each fold. Inside each entry of the dictionary, a list of paths, referencing the image that belongs to said set in that given fold of the crossvalidation. An example of this json file can be found in: "fold_data_example.json"

This project requires three sets of images:

- The original OCT images, without any alteration.
- Binary images, where the microcystic regions are marked in white and the non-relevant regions in black (i.e. binary masks)
- ROI binary images, where the region of interest for the second stage of the experiment is denoted (for example, a morphological dilation version of the masks with a window of 64x64)

In the code, the configuration of the dataset can be set as follows:
```
##################################
fold_info = 'FOLD_DATA_FN_L_6V2T1'
p_images = './dataset/ORIGINALS/'
p_labels = './dataset/MICROCYSTS/'
p_roi = './dataset/SUBMASKS_64/'
##################################
```
"fold_info" corresponds to the JSON file with the cross-validation fold information.
"p_images" is the path to the image dataset
"p_labels" is the path to the binary masks that contain the regions with microcysts.
"p_roi" is the path to the dilated binary masks that represent the valid region from to which extract positive samples in the coarse surrounding region.

Secondly, the training configuration can be set in the "c" variable as desired:

```
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

Finally, "window_ranges" allows to configure a list of window ranges to study in that given stage of the proposed methodology (Stage 1 or 2).

"ablation_main" will generate automatically a file ending with "_conf" to store the experiment name configuration for reproducibility, as well as folders with the results of training.

In folder '''maps''' we include the code to generate maps with the given folders.

There are two models included, '''model_30.rar''' and ''model_64.rar'''. The 64 model corresponds to the model trained to extract the coarse region of interest. The 30 model, the precise microcysts inside that region.

### Map generation

To generate the maps, execute:

'''python generateMicrocystRepresentation.py ./target_image.png ./roi.png ./generated_map.png'''

Where ./target_image.png is the original OCT image and ./roi.png represents the retinal region of interest to be analyzed by the models. The map will be saved as the third argument, presented here as "./generated_map.png".

## 3. Permanent Links

To enhance reproducibility, the following resources have been archived using Zenodo for persistent access:

https://doi.org/10.5281/zenodo.14006681

## 4. Usage Disclaimer and Citation

This code is intended for statistical purposes only, and no private data or fees are required. If you use this image set in your work, please include the following reference:

P. L. Vidal, J. de Moura, J. Novo, M. Ortega, "Robust fully-automatic multi-stage learning for the intuitive representation of MME accumulations in OCT images", The Visual Computer, 2024 (pending acceptance).

