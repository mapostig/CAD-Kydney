# Object detection and Multi-label LOCAL Image Classification

Object detection provides labels that locate the pathology in the kidney US image. To target the detection task we applied the most common
object detection architecture, known as **Faster R-CNN**. 

In addition to the local detection, a global image-level diagnosisscore is proposed by applying an aggregation mechanism.
The goal of this step is therefore to aggregate, for every image, the scores of the detected bounding boxes into
a vector, providing a global image classification.

* **Finetuning.py**: main workflow. Create datasets and dataloaders, load the pretrained model, training and saving the trained model.
* **CustomDataset.py**: Custom dataset for object detection, with the images, labels and bounding boxes. Contains the dataload functions.
* **CustomTRansforms.py**: Custom tranforms for the given dataset. Goal--> data augmentation
* **EarlyStopping.py:** Earlystopping implementation
* **bbox_utils.py:** some functions regarding the bounding boxes. Transforms and visualization.
* **Evaluation.py:** methods regarding the evaluation. Calculate IoU and evaluate the trained model.
* **utils.py:** training methods from Object Detection with Pytorch (tutorial) adapted to the given problem and visualizations.
* **evaluate_results.py:** different functions that creates the files containing the detection results for further evaluation.
* **VisualizeResults.py:** Loading the saved results in order to visualize the predictions and obtain certain evaluation metrics for the given classes.


Remark: To see the mAP use https://github.com/Cartucho/mAP from the output generated by create_detection_results_files function in  **evaluate_results.py**
