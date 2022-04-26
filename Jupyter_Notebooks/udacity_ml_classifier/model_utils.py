# Imports python modules
import torch
from torch import nn, optim
from torchvision import models
from classifier import Classifier
from os_utils import check_path, make_dir, quit

def new_model(hidden_units, model_name, drop_rate=0.2):
    """ Build new model
        Returns model
    """
    model = getattr(models, model_name)(pretrained=True)
    
    # get maximum number of input units for model
    try:
        # if pretrained model has prebuilt classifier
        # get input units from prebuilt classifier
        if isinstance(model.classifier, nn.Linear):
            input_units = model.classifier.in_features
        else:
            for i in model.classifier:
                if isinstance(i, nn.Linear):
                    input_units = i.in_features
                    break
    except:
        # if pretrained model has no classifier
        # set input units from fc layer
        input_units = model.fc.in_features
    
    output_units = 102
    
    # if hidden unit sizes specified
    if hidden_units:
        # ensure that hidden unit sizes are arranged in descending order
        hidden_units = sorted(hidden_units, reverse=True)
    # if hidden unit sizes not specified
    # create list with a single integer value evenly spaced between input
    # unit sizes and output unit sizes
    else:
        hidden_units = [input_units-int((input_units - output_units)/2)]
    
    for param in model.parameters():
        param.requires_grad = False

    # build own classifier
    classifier = Classifier(input_units, hidden_units, output_units, drop_rate)

    # replace model classifier with own classifier
    if 'classifier' in dir(model):
        model.classifier = classifier
    else:
        model.fc = classifier
    
    return model

def load_model(filepath):
    ''' Loads checkpoint for the model and rebuilds,
        returns the rebuilt model
    '''
    # check that file exists
    filepath, exists = check_path(filepath)
    
    if not exists:
        quit("{} does not exist".format(filepath))
    
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['model_arch'])(pretrained=True)
    
    if 'classifier' in checkpoint:
        model.classifier = checkpoint['classifier']
    else:
        model.fc = checkpoint['fc']
    
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.image_dir = checkpoint['image_dir']
    
    return model

def load_optimizer(filepath):
    ''' Loads checkpoint for optimizer,
        returns the optimizer to stored state
    '''
    # check that file exists
    filepath, exists = check_path(filepath)
    
    if not exists:
        quit("{} does not exist".format(filepath))
    
    checkpoint = torch.load(filepath)
    
    if 'classifier' in checkpoint:
        params = checkpoint['classifier'].parameters()
    else:
        params = checkpoint['fc'].parameters()
    
    optimizer = getattr(optim, checkpoint['optim'])(params, 
                                                    checkpoint['learn_rate'])
    optimizer.load_state_dict(checkpoint['optim_dict'])
    
    return optimizer

def save_model(args, model, optimizer, class_to_idx):
    
    save_dir = args.save_dir
    # check that directory exists and remove slash at end if needed
    save_dir, exists = check_path(save_dir)

    # if specified save directory does not exist,
    # create directory using specified name
    if not exists:
        make_dir(save_dir)

    # Create checkpoint and save 
    model.class_to_idx = class_to_idx
    
    if 'classifier' in dir(model):
        inputs = model.classifier.hidden[0].in_features
        outputs = model.classifier.output.out_features
        classifier = 'classifier'
        model_classifier = model.classifier
    else:
        inputs = model.fc.hidden[0].in_features
        outputs = model.fc.output.out_features
        classifier = 'fc'
        model_classifier = model.fc

    checkpoint = {'input_size': inputs,
                  'output_size': outputs,
                  'model_arch': args.arch,
                  classifier : model_classifier,
                  'epochs': args.epochs,
                  'optim_arch': args.optim,
                  'optim_dict': optimizer.state_dict(),
                  'learn_rate': args.learning_rate,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'image_dir': args.data_directory}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')