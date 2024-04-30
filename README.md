# CSA-Net
Official PyTorch implementation of: 

[A Flexible 2.5D Medical Image Segmentation Approach with In-Slice and Cross-Slice Attention](https://arxiv.org/pdf/..)

This is a 2.5D Cross-Slice and In-Slice attention-based transformer model for 2.5D MRI Dataset.

The code is only for research purposes. If you have any questions regarding how to use this code, feel free to contact Amarjeet Kumar (amarjeetkumar@ufl.edu).

## Requirements
Python==3.9.16
* torch==1.10.1
* torchvision==0.11.2
* numpy
* opencv-python
* tqdm
* tensorboard
* tensorboardX
* ml-collections
* medpy
* SimpleITK
* scipy
* `pip install -r requirements.txt`

## Dataset
- ProstateX Dataset :- 
- Dataset can be accessed here: [official ProstateX website](https://www.cancerimagingarchive.net/collection/prostatex/)

## Usage

### 1. Download Google pre-trained ViT models
*[Get models in this link](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false) : R50-ViT-B_16
* Save your model into folder "model/vit_checkpoint/imagenet21k/".
```bash
../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

1. Access to the ProstateX dataset:
   Sign up in the [official ProstateX website](https://www.cancerimagingarchive.net/collection/prostatex/) and download the dataset. Partition it in training and testing dataset as :- trainVol, trainMask, testVol, testMask. Put these folders under data directory.
      * Run the preprocessing script, which would generate train_npz folder containing 2D images in folder "data/", data list files in folder "lists/" and train.csv for overview.
```
python preprocessing.py
```
OR You can download the preprocessed dataset using this [link](https://drive.google.com/drive/folders/1qAkX34E_5kP-2pKDI0RChqWKfTNl1FVQ?usp=sharing). After downloading, copy  "list" directory from utils to "/CSANet" path to store text files containing the names of all samples for each task.

The directory structure of the whole project is as follows:

```bash
.
├── CSANet
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           └── R50+ViT-B_16.npz
│           
└── data
    ├── trainVol
    ├── trainMask
    └── testVol
    ├── testMask
    └── train_npz     
```


Note:- You can also directly utilize lists folder from utils to handle class imbalance issue for the ProstateX dataset.

### 3. Train/Test
* Please go to the folder "CSANet/" and it's ready for you to train and test the model.
```
python train.py
python test.py
```
You can see the test outputs in the "Results/" folder.

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
