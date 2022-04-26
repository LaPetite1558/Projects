# Imports python modules
import argparse
from torch import nn, optim
from torchvision import models

def get_train_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run train.py from a terminal window.
    Command Line Arguments:
      1. Image Folder as data_directory with default value 'flowers'
      2. Checkpoint Folder as --save_dir
      3. Model Architecture as --arch with default value 'vgg16'
      4. Optimizer Algorithm as --optim with default value 'SGD'
      5. Loss Function as --loss with defailt value 'NLLLoss'
      6. Learning Rate as --learning_rate with default value 0.1
      7. Number of Hidden Units as --hidden_units with default value [512]
      8. Drop Rate as --drop_rate
      9. Number of Epochs to train as --epochs with default value 10
      10. Use GPU to train as --gpu
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments
    parser.add_argument('data_directory', type = str, nargs='?',
                        default = 'flowers',
                        help = 'path to the folder of images')
    parser.add_argument('--save_dir', type = str, 
                        help = 'path to the folder of checkpoint')
    parser.add_argument('--arch', type = str, default = 'vgg16',
                        choices=[m for m in dir(models) if 'net' in m 
                                 or 'vgg1' in m],
                        help = 'name of CNN model architecture to use')
    parser.add_argument('--optim', type = str, default = 'SGD',
                        choices=[o for o in dir(optim) if o != 'Optimizer' 
                                 and o[0].isupper()],
                        help = 'name of Optimizer Algorithm to use')
    parser.add_argument('--loss', type = str, default = 'NLLLoss',
                        choices=[l for l in dir(nn.modules.loss) if 'Loss' in l 
                                 and l[0] != '_'],
                        help = 'name of Loss Function to use')
    parser.add_argument('--learning_rate', type = float, default = 0.1,
                        help = 'learning rate for the optimizer')
    parser.add_argument('--hidden_units', type= int, nargs = '+',
                        help = 'hidden unit input sizes')
    parser.add_argument('--drop_rate', type = float, 
                        help = 'drop rate for the classifier')
    parser.add_argument('--epochs', type = int, default = 10,
                        help = 'number of epochs to train')
    parser.add_argument('--gpu', action='store_true',
                        help = 'train with gpu or not')
    
    return parser.parse_args()


def get_predict_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run predict.py from a terminal window.
    Command Line Arguments:
      1. Image to predict as img_path with default value 'flowers/test/10/image_07090.jpg'
      2. Checkpoint as checkpoint with default value 'save/checkpoint.pth'
      3. Top K most likely classes as --top_k with default value 5
      4. File of category names as --category_names
      5. Return actual class as --actual
      6. Use GPU to train as --gpu
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments
    parser.add_argument('img_path', type = str, nargs='?',
                        default = 'flowers/test/10/image_07090.jpg',
                        help = 'path to the image to predict')
    parser.add_argument('checkpoint', type = str, nargs='?',
                        default = 'save/checkpoint.pth',
                        help = 'path to the model checkpoint')
    parser.add_argument('--top_k', type = int, default = 5,
                        help = 'number of top most likely classes')
    parser.add_argument('--category_names', type = str,
                        help = 'path to file of real category names')
    parser.add_argument('--actual', action='store_true',
                        help = 'show actual class')
    parser.add_argument('--gpu', action='store_true',
                        help = 'predict with gpu or not')
    
    return parser.parse_args()