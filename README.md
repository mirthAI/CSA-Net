# CSA-Net
Official PyTorch implementation of: 

[A Flexible 2.5D Medical Image Segmentation Approach with In-Slice and Cross-Slice Attention](https://arxiv.org/abs/2405.00130)

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

## Usage

### 1. Download Google pre-trained ViT models
*[Get pretrained vision transformer model using this link](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false) : R50-ViT-B_16
* Save your model into folder "model/vit_checkpoint/imagenet21k/".
```bash
../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Here we use the ProstateX dataset as an example. This dataset consists of T2-weighted prostate MRI, along with expert-annotation of four different prostate regions: transition zone, peripheral zone, urethra, and anterior fibromuscular stroma.

1. Access to the ProstateX dataset:
   Sign up in the [official ProstateX website](https://www.cancerimagingarchive.net/collection/prostatex/) and download the dataset. Partition it in training and testing dataset as :- trainVol, trainMask, testVol, testMask. Put these folders under data directory.
      * Run the preprocessing script, which would generate train_npz folder containing 2D images in folder "data/", data list files in folder "lists/" and train.csv for overview.
```
python preprocessing.py
```
OR You can download our preprocessed dataset using this [link](https://drive.google.com/drive/folders/1qAkX34E_5kP-2pKDI0RChqWKfTNl1FVQ?usp=sharing). After downloading, copy the "lists" directory from utils to "/CSANet" path to store text files containing the names of all samples for each task.

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

## Citations
Kindly cite our paper as follows if you use our code.
```bibtex
@misc{sivaraman2024retinaregnet,
    title={A Flexible 2.5D Medical Image Segmentation Approach with In-Slice and Cross-Slice Attention},
    author={Amarjeet Kumar and Hongxu Jiang and Muhammad Imran and Cyndi Valdes and Gabriela Leon and Dahyun Kang and  Parvathi Nataraj and Yuyin Zhou and Michael D. Weiss and Wei Shao},
    year={2024},
    eprint={2405.00130},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
