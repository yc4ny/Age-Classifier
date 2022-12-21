import torch 
import torch.nn as nn
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import matplotlib.pyplot as plt
from torchvision import transforms, models
import numpy as np
import argparse
import os
import cv2 

parser = argparse.ArgumentParser(description = "Age Classification")
parser.add_argument('--input', type = str, default = 'testing/input_dir', help = 'Inference Directory')
parser.add_argument('--output', type = str, default = 'testing/output_dir', help = 'Output Directory')
parser.add_argument('--checkpoint', type = str, default = 'checkpoints/best_checkpoint.pth', help = 'Model Path')
args = parser.parse_args()

label_to_age = {
    0: "0-6 years old",
    1: "7-12 years old",
    2: "13-19 years old",
    3: "20-30 years old",
    4: "31-45 years old",
    5: "46-55 years old",
    6: "56-66 years old",
    7: "67-80 years old"
}

test_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

def imshow(input):
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)
    plt.show()

model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 8) # transfer learning
model = model.cuda()
model_path = args.checkpoint
model.load_state_dict(torch.load(model_path)) # load model
model.eval() # set model to evaluation mode

directory = args.input

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    img = Image.open(f).convert('RGB')
    img = test_transform(img).unsqueeze(dim = 0).cuda() # (1, 3, 128, 128)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        txt = "Predicted Age: " + label_to_age[preds[0].item()]
        img = cv2.imread(f)
        img = cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.output, filename), img)


