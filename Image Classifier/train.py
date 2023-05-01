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


def data_loader(data_dir):
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test' #TODO Remove test dir and loaders
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = tv.transforms.Compose([tv.transforms.CenterCrop(224),
                                              tv.transforms.Resize((224, 224)),
                                              tv.transforms.RandomRotation(30),
                                              tv.transforms.RandomHorizontalFlip(p=0.5),
                                              tv.transforms.ToTensor(),
                                              tv.transforms.Normalize(
                                                  mean=[0.485, 0.456, 0.406], 
                                                  std=[0.229, 0.224, 0.225])])

    val_and_test_transforms = tv.transforms.Compose([tv.transforms.CenterCrop(224),
                                                     tv.transforms.Resize((224, 224)),
                                                     tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(
                                                         mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = [ImageFolder(train_dir, transform=train_transforms),
                      ImageFolder(valid_dir, transform=val_and_test_transforms),
                      ImageFolder(test_dir, transform=val_and_test_transforms)]

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=32, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=32, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=32, shuffle=True)]
    

        
    return dataloaders, image_datasets
        
        
def build_model(arch, hidden_units):
    
    # check which arch to use
    if arch=='vgg16':
        model = models.vgg16(pretrained=True)
    elif arch=='vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
        
    # check hidden units arg availability
    if hidden_units:
        hidden_units = hidden_units
    else:
        hidden_units = 4096
        
    hidden_units = int(hidden_units)
    
    # freez params
    for param in model.parameters():
        param.requires_grad=False
        
    # new classifier
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 512)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    return model
    
def train_model(dataloaders, device, model, epochs, learning_rate):
    
    epochs = int(epochs)
    learning_rate = float(learning_rate)
    steps = 0
    print_seq = 15
    
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    train_losses, valid_losses = [], []

    for e in range(epochs):
        running_loss = 0
        train_loss = 0 

        for images, labels in dataloaders[0]:
            steps+=1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model.forward(images)
            train_loss = criterion(log_ps, labels)
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()


            if steps % print_seq == 0:
                accuracy = 0
                valid_loss = 0

                print(f'Epoch: {e+1} Step: {steps}')
                print('----------')

                model.eval()

                with torch.no_grad():
                    for images, labels in dataloaders[1]:
                        images, labels = images.to(device), labels.to(device)


                        log_ps = model.forward(images)
                        cb_loss = criterion(log_ps, labels)

                        valid_loss += cb_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                if steps % print_seq == 0 or steps == 1 or steps == len(dataloaders[0]):
                    print_loss = round(running_loss/print_seq, 3) 
                    print_val_loss = round(valid_loss/len(dataloaders[1]), 3)
                    print_accuracy = round(accuracy/len(dataloaders[1]), 3)

                    train_losses.append(print_loss)
                    valid_losses.append(print_val_loss)

                    print(f'Training Loss: {print_loss} \t Validation Loss: {print_val_loss} \t Accuracy: {print_accuracy}')
                    print('----------')

                    running_loss = 0
                    
                    
def save_model(save_dir, image_datasets, model, epochs, arch):
    #Save the checkpoint 
    model.class_to_idx = image_datasets[0].class_to_idx

    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'arch': arch,
                  'input_size': 25088,
                  'output_size': 102,
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))



def main():
    
    print('Training Started...\n')
    
    # parsing arguments
    parser = argparse.ArgumentParser(description='Model training program')

    parser.add_argument('--data_dir', dest='data_dir', required=True, help='Path of data directory.')
    parser.add_argument('--save_dir', dest='save_dir', help='Set directory to save checkpoints.')
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg13', 'vgg16', 'vgg19'], help='Select the architecture.')
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001', help='Set the learning rate.')
    parser.add_argument('--hidden_units', dest='hidden_units', default='4096', help='Set hidden units.')
    parser.add_argument('--epochs', dest='epochs', default='5', help='Set epochs.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU.')
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu
    
    # check GPU arg
    if gpu==True:
        device = 'cuda'
        print('Using CUDA\n')
    else:
        device = 'cpu'
        print('Using CPU\n')
        
    # print args
    print(f'Architecture: {arch}')
    print(f'Learning Rate: {learning_rate}')
    print(f'Hidden Layer: {hidden_units}')
    print(f'Epochs: {epochs}')
    print(f'Data Directory: {data_dir}')
    print(f'Save Directory: {save_dir}\n')
        
    # check checkpoint path availability
    if save_dir==None:
        save_dir = 'saved_models'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    dataloader, image_datasets = data_loader(data_dir)
    model = build_model(arch, hidden_units)
    train_model(dataloader, device, model, epochs, learning_rate)
    save_model(save_dir, image_datasets, model, epochs, arch)
    

if __name__ == '__main__':
    main()
