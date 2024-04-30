# CSA-Net

This repo holds code for [A Flexible 2.5D Medical Image Segmentation Approach with In-Slice and Cross-Slice Attention](https://arxiv.org/pdf/..)
 
## Usage

### 1. Download Google pre-trained ViT models
[Get models in this link](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false) : R50-ViT-B_16
* Put the downloaded model file in 
```bash
../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

1. Access to the ProstateX dataset:
   1. Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/) and download the dataset. Convert them to numpy format, normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume for testing cases.
   2.  You can also directly use preprocesses dataset from this [link] ([https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false](https://drive.google.com/drive/folders/1qAkX34E_5kP-2pKDI0RChqWKfTNl1FVQ?usp=sharing)
The directory structure of the whole project is as follows:

```bash
.
├── CSANet
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           ├── R50+ViT-B_16.npz
│           
└── data
    ├── trainVol
    ├── trainMask
    └── testVol
    ├── testMask
    ├── train_npz     
```

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test
- Run the setup script. This will prepare the 2D training dataset from 3D Volumes and create the lists for training and testing data.
```bash
python setup.py
cd CSANet/
```

- Run the train script.

```bash
python train.py
```

- Run the test script. It supports testing on 3D volumes.

```bash
python test.py
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
