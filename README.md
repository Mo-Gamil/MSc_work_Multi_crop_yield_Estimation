# MSc_work_Multi_crop_yield_Estimation
This repository includes all the developed codes for the MSc work of multi-crop yield estimation. It contains the data preparation, downloading and processing pipelines. Furthermore, it contains all the code of the developed DL models. Those DL models are:
1- Base CNN for multi-crop yield estiamation using 8 bands of sentinel-2 images
2- Base CNN for multi-crop yield estiamation using 8 bands of sentinel-2 images. Additionally the Cropland Data Layer (CDL) is added as a 9th band.
3- U-net multi-task model for crop type identification and yield estimation
4 - Swin multi-task model for crop type identification and yield estimation

![image](https://github.com/user-attachments/assets/9930a556-eed5-4cc5-9b26-e030cd7a90ea)
Figure 12: The architecture of the two CNNs models (with CDL and without CDL)
![image](https://github.com/user-attachments/assets/db8821c0-fcf7-4211-8b1b-34b75037bbc0)
Figure 13: The architecture of the multi-task learning U-Net model

![image](https://github.com/user-attachments/assets/aabbe408-151e-49c9-b4f2-d6946e68b40b)
Figure 14: The architecture of the multi-task learning Swin model
