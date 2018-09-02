import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.models import vgg19, densenet121, vgg16
from torchvision import datasets, models, transforms
import torchvision
from torch import nn, optim
import torch
import torch.nn.functional as F
from collections import OrderedDict
import json
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Predict the type of a flower')
parser.add_argument('--checkpoint', type=str, help='Path to checkpoint' , default='checkpoint.pth')
parser.add_argument('--image_path', type=str, help='Path to file' , default='flowers/test/28/image_05230.jpg')
parser.add_argument('--gpu', type=bool, default=True, help='Whether to use GPU during inference or not')
parser.add_argument('--topk', type=int, help='Number of k to predict' , default=0)
parser.add_argument('--cat_to_name_json', type=str, help='Json file to load for class values to name conversion' , default='cat_to_name.json')
args = parser.parse_args()


with open(args.cat_to_name_json, 'r') as f:
    cat_to_name = json.load(f)
image_path = args.image_path
device = 'cuda' if args.gpu else 'cpu'

# : Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(checkpoint):
    checkpoint = torch.load(args.checkpoint)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    for param in model.parameters():
        param.requires_grad = False
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    return model, checkpoint

model, checkpoint = load_checkpoint(args.checkpoint)

def process_image(image):
    image = image.resize((round(256*image.size[0]/image.size[1]) if image.size[0]>image.size[1] else 256,
                          round(256*image.size[1]/image.size[0]) if image.size[1]>image.size[0] else 256))  
    
    image = image.crop((image.size[0]/2-224/2, image.size[1]/2-224/2, image.size[0]/2+224/2, image.size[1]/2+224/2))

    np_image = (np.array(image)/255-[0.485,0.456,0.406])/[0.229, 0.224, 0.225]
    np_image = np_image.transpose((2,0,1))
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    return torch.from_numpy(np_image)

# : Process a PIL image for use in a PyTorch model    
im = Image.open(image_path)
processed_im = process_image(im)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # : Implement the code to predict the class from an image file
    im = Image.open(image_path)
    processed_im = process_image(im).unsqueeze(0)
    model.to(device)
    model.eval()    
    with torch.no_grad():
        processed_im = processed_im.to(device).float()
        output = model(processed_im)
        ps = torch.exp(output)
    pred = ps.topk(topk)
    flower_ids = pred[1][0].to('cpu')
    flower_ids = torch.Tensor.numpy(flower_ids)
    probs = pred[0][0].to('cpu')
    idx_to_class = {k:v for v,k in checkpoint['class_to_idx'].items()}
    flower_names = np.array([cat_to_name[idx_to_class[x]] for x in flower_ids])
        
    return probs, flower_names

if args.topk:
    probs, flower_names = predict(image_path, model, args.topk)
    print('Probabilities of top {} flowers:'.format(args.topk))
    for i in range(args.topk):
        print('{} : {:.2f}'.format(flower_names[i],probs[i]))
else:
    probs, flower_names = predict(image_path, model)
    print('Flower is predicted to be {} with {:.2f} probability'.format(flower_names[0], probs[0]))