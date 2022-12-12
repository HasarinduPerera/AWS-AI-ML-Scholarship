# Imports here
import argparse
import torch
import numpy as np
import torchvision as tv
from torchvision.datasets import ImageFolder
import torchvision.models as models
from collections import OrderedDict
from torch import nn, optim
from PIL import Image
import json
import os


def load_checkpoint(filepath):
    # load checkpoint
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    
    
    # check model arch and load model
    if arch=='vgg16':
        model = tv.models.vgg16(pretrained=True)
    elif arch=='vgg13':
        model = tv.models.vgg13(pretrained=True)
    else:
        model = tv.models.vgg19(pretrained=True)
        
    
    # freez params
    for param in model.parameters():
        param.requires_grad=False
       
    # get model params
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # process image
    image_pil = Image.open(image)
    img_processing = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    
    image = img_processing(image_pil)
    
    return image


def predict(image_path, model, topk, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = image.numpy()
    image = torch.from_numpy(np.array([image])).float()
    
    model.to(device)
    image = image.to(device)
    model.eval()
    
    # Parse topk to int
    topk = int(topk)
    
    with torch.no_grad():
        prediction = model.forward(image)
        prob = torch.exp(prediction)
        
        top_probs, top_labels = prob.topk(topk)
        
        #print(prob.topk(topk))
    
        top_probs = np.array(top_probs.detach())[0]
        top_labels = np.array(top_labels.detach())[0]

        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_classes = [idx_to_class[lab] for lab in top_labels]
        
        # check name file availability
        if not cat_to_name==None:
            with open('cat_to_name.json', 'r') as f:
                cat_to_name = json.load(f)
                top_names = [cat_to_name[lab] for lab in top_classes]
        else:
            top_classes = None
        
        # print results
        print(f'Top Class(es): \t\t{top_classes}')
        print(f'Top Prob(s): \t\t{top_probs}')
        print(f'Top Class(es) Name(s): \t{top_names}')

        #return top_probs, top_classes
    
    


def main():
    print('Inference Started...\n')
    
    # parsing arguments
    parser = argparse.ArgumentParser(description='Model training program')

    parser.add_argument('--img_path', dest='img_path', required=True, help='Path of the image.')
    parser.add_argument('--checkpoint', dest='checkpoint', required=True, help='Select the saved model checkpoint.')
    parser.add_argument('--top_k', dest='top_k', default='1', help='Return top K most likely classes.')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json', help='Use a mapping of categories to real names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU.')
    
    args = parser.parse_args()
    
    img_path = args.img_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu
    
    # check GPU arg
    if gpu==True:
        device = 'cuda'
        print('Using CUDA\n')
    else:
        device = 'cpu'
        print('Using CPU\n')
        
    # print args
    print(f'Image Path: {img_path}')
    print(f'Checkpoint: {checkpoint}')
    print(f'Top K: {top_k}')
    print(f'Category Names File: {category_names}\n')

    
    model = load_checkpoint(checkpoint)
    image = img_path

    predict(image, model, top_k, device, category_names)




if __name__ == '__main__':
    main()
