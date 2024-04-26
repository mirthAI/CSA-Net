# CSA-Net

This repo holds code for [A Flexible 2.5D Medical Image Segmentation Approach with In-Slice and Cross-Slice Attention](https://arxiv.org/pdf/..)

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link] (https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)): R50-ViT-B_16
* Put the downloaded model file in 
```bash
../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets/README.md) for details.

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
cd CSANet/
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
