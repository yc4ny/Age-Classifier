# Age Classifier <br/> 
## Model to estimate age of people built using PyTorch<br/> 

<br/>
<p align= "center">
<img src="repo_img/output.png" width="600" height="450" />
</p>
<br/>

> - Predicted age of some famous stars. 
> - Input: Any size, type (extension) facial RGB image of a person. 
> - Output: Predicted Age Range
> - 70.3215 % accuracy achieved. 

## Environment Setups
> Note: This code was developed on Ubuntu 20.04 with Python 3.7. Later versions should work, but have not been tested.
Create and activate a virtual environment to work in, e.g. using Conda:

```
conda create -n venv_age python=3.7
conda activate venv_age
```

> Install Pytorch according to your GPU verison. eg) RTX 3090, CUDA 11.0 
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
>Install other dependencies. 
```
pip install -r requirements.txt
```
### Dataset Setup
This project uses the ```Facial Image Data with Known Family Relationships``` dataset from AI Hub. 
Please download the dataset from:
```https://drive.google.com/file/d/1YOfavMsYwv21IQ19iDN3THppfhx4cM3n/view?usp=share_link```

### Folder Hierarchy
Once you sucessfully downloaded and unzips dataset files, you should have a directory similar to this:
   ```
    ./face_dataset
    ├── 
    │   └── fixed_test_val
    │   └── fixed_val_dataset
    |   └── test_images
    │   └── train_images
    │   └── val_images
    |   └── custom_test_dataset.csv
    |   └── custom_train_dataset.csv
    |   └── custom_val_dataset.csv
   ```
## Running the demo
To run inference on the pretrained model, run:
```
python demo.py
```
#### Model options
```
  --input               STR    Input graph path.                                    
```
> - To use your own set of images, place your images in the ```input``` folder. <br/>
> - Update your camera parameters ```K``` in the ```demo.py``` file. Accurate camera parameters are needed to produce robust results. 
> - Each step is printed when an input image has been transformed and projected on to the new image plane. 




