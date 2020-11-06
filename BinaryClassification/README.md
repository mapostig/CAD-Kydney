This folder contains the code for the binary classification of kidney images as healthy or pathological.


* **Custom_dataset.py**: the kidney dataset class
* **Custom_trasnforms.py**: contains the different transforms applied to the dataset
* **Data_load_functions.py**: how the images, labels and annotations are loaded into the dataset
* **Early_stopping.py**: an early stopping implementation
* **Test_models.py**: contains all the workflow (main). The images are loaded, the model is trained and evaluates the performance over the test set
* **ROC_analysis.py**: performs different measures on the obtained outputs
