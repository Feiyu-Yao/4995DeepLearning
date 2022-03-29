# 4995DeepLearning Competition Code

## Technology:

<img src="/1648523447253.jpg" width="25%">   



Fine-tune: The pretrained model can be a good start point. Dehaze model shares the same low level feature. Freeze VGG layer for AECR-NET. Freeze no layers for YOLOv5.

## Dependencies

Basically this work is done in two virtual machines. Thus different dependencies is needed:



# Dependencies 1
* Python3.7
* CUDA 10.2
* torch 1.4.0
* torchvision 0.5.0
* tqdm
* tensorflow
* pillow
# Dependencies 2
* python3
* torch>=1.7.0
* torchvision>=0.8.1
* tqdm>=4.41.0


# Dehaze AECR-NET
1: Install DCNv2:
```
cd DCNc2
bash ./make.sh
```
2: Install requirements
```
pip install -r requirement.txt

```
3: Train
```
python trainin.py
```
Referring for options if you want to change specific terms.

4: Test
```
python test.py 
```
# YOLOv5
1: Install requirements
```
pip install -r requirement.txt
```
2: Install wandb
```
pip install wandb
```
3: Create dataset.yaml for A2I2 dataset 

```
path: Dataset/dataset-vehicles  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: [ 'Vehicle']  # class names
```

4: Fine-tuning on a pre-trained model of yolov5.
```
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5m.pt
```
 
5: After train, gives you weights of train and you should use them for test.
```
python detect.py --weights runs/train/exp1/weights/best.pt --source test_images/
```
