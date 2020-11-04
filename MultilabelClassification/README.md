
# Multi-label Global Image Classification
This folder contains the code for the Multi-label Global Image Classification.
The labels are binary vectors according to the present/absense of a given class
    class_names = ["Healthy", "Cyst", "Pyramid", "Hydronephrosis", "Others",
            "Poor Corticomedular Differenciation", "Hyperechogenic cortex"]
    label example: [0,1,1,1,0,0,1]


* **Custom_dataset.py**: the kidney dataset class. Each item is a dictionary containing the image, the kidney segmentation mask and the label vector
* **Custom_trasnforms.py**: contains the different transforms applied to the dataset
* **Data_load_functions.py**: how the images, labels and masks are loaded into the dataset
* **Early_stopping.py**: an early stopping implementation
* **Utils.py**: contains the dataloaders generation, training and evaluation methods
* **Main.py**: contains all the workflow. The images are loaded, the model is trained and evaluates the performance over the test set
* **evaluate_results.py**: write the ground truth and the predictions from the multi-label classification into .txt files for evaluation
