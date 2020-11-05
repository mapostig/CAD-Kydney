# Automatic Diagnosis Of Renal Pathologies Using Kidney Ultrasound Imaging

**Author: María Postigo Fliquete**
## **Please reference this source if you employ the CAD code and implementations**

The identification of renal pathologies on their early stages is essential to avoid the development of the disease. Therefore, good primary care judgment is critical. However, the lack of skill in adquiring and interpreting kidney US images in primary care leads to significant challenges in clinical decision making.

In this paper we propose a CAD system for kidney ultrasound images, which may help both primary care and specialist nephrologists, acting as a "second opinion". Within the different approaches proposed, the combination of a global and a local image classification results in an **AUC = 0.9017** with a **SP-95 = 0.4911**. This is a high confidence CAD system that can be exploited in primary care. On the assumption that every symptomatic patient is referred to an specialist, with this system, 1 out of 2 healthy patients referrals could be avoided. 

In addition to prevent the patients' overload for the specialists, the proposed CAD system is a good multi-pathology classifier that can help nephrologists in decision making, with a multi-label **AUC = 0.8366**.

**This is my Master Thesis in "Universidad Carlos III de Madrid"**

Program: MASTER IN INFORMATION HEALTH ENGINEERING

**Score: 96/100**

The paper is available in "TFM_MariaPostigo.pdf"

The presentation is available in "TFM_MariaPostigo_Slides.pdf"

CODE:
* Binary classification
* Multilabel classification (gobal image classification)
* Multilabel classification (Based on local object detection)

Examples of images, annotations, labels, splits and masks will be available soon.
