# Imports python modules
import torch
from torch import nn, optim
from get_args import get_train_args
from preprocess import load_data
from model_utils import new_model, save_model
from os_utils import quit

"""
Usage: python train.py [arguments]
arguments:
    1. data_directory - if given, requires a valid directory to be specified
                        if not given, will use default value 'flowers'
    2. --save_dir - optional, if given, will save a checkpoint to specified 
                    directory, otherwise will not save a checkpoint
    3. --arch - optional, if given, will build model using specified model
                architecture, otherwise will use default vgg16 model.
                if invalid model architecture is provided, command line will
                show error message
    4. --optim - optional, if given, will create optimizer using specified algorithm,
                 otherwise will use default SGD algorithm.
                 if invalid algorithm name is provided, command line will
                 show error message
    5. --loss - optional, if given, will create criterion using specified loss
                function, otherwise will use default NLLLoss function.
                if invalid loss function is provided, command line will
                show error message
    6. --learning_rate - optional, if given, will use specified value,
                         otherwise will use default value 0.1
    7. --hidden_units - optional, if given, will use specified values as
                        inputs and outputs to create hidden layers, otherwise
                        will use default value [512] to create 1 hidden layer
                        with 512 output units
    8. --drop_rate - optional, if given, will use specified value for drop rate, 
                     otherwise Classifier will use own default value 0.2
    9. --epochs - optional, if given, will use specified value for epochs to train, 
                  otherwise will use default value 10
    10. --gpu - optional, if given, will use GPU if available, if not given, will
                default to CPU
"""

def validate(model, criterion, validloader, device):
    valid_loss = 0
    accuracy = 0

    for data, labels in validloader:
        data, labels = data.to(device), labels.to(device)
        logps = model.forward(data)
        batch_loss = criterion(logps, labels)

        valid_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    return valid_loss, accuracy 
    
def train(model, epochs, criterion, optimizer, trainloader, validloader, device):
    steps = 0
    train_loss = 0
    print_every = 40

    for e in range(epochs):
        model.train()

        for data, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(data)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():    
                    valid_loss, accuracy = validate(model, criterion, validloader, device)

                print("Epoch: {}/{}.. Training loss: {:.2f}  Validation loss: {:.2f}  Accuracy: {:.2f}%".format(
                    e+1, epochs, train_loss/print_every, valid_loss/len(validloader), 
                    accuracy/len(validloader) * 100))

                train_loss = 0
                model.train()

def main():
    ''' Trains the PyTorch model according to command line input arguments
        If save directory is given, saves model checkpoint
    '''
    # get input arguments for training
    args = get_train_args()
    
    # if user asks for gpu, enable gpu if available
    if args.gpu and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device(0)
    else:
        print("GPU unavailable. Defaulting to CPU...")
        device = torch.device("cpu")
        
    # if drop rate for model was provided
    if args.drop_rate:
        model = new_model(args.hidden_units, args.arch, args.drop_rate)
    else:
        model = new_model(args.hidden_units, args.arch)

    # get criterion from loss function
    criterion = getattr(nn, args.loss)()
    
    # if model has a classifier object get parameters
    if 'classifier' in dir(model):
        params = model.classifier.parameters()
        to_print = model.classifier
    # if model does not have classifier object
    # get parameters from fc layer
    else:
        params = model.fc.parameters()
        to_print = model.fc
        
    # create optimizer using model parameters and specified learning rate
    optimizer = getattr(optim, args.optim)(params, args.learning_rate)

    model.to(device);
    
    print("Using {} model".format(args.arch))
    print("Using {} as loss function".format(args.loss))
    print("Using {} as optimizer, lr={}".format(args.optim, args.learning_rate))
    print(to_print)
    
    usr_input = input("Model ready to train for {} epochs. Proceed? [y|n] ".format(args.epochs))
    if usr_input != 'y':
        quit("Exiting application...")
    
    # load data
    trainloader = load_data(args.data_directory + '/train', 
                            train=True)
    validloader = load_data(args.data_directory + '/valid')
    
    train(model, args.epochs, criterion, optimizer, trainloader, validloader, device)
    
    # if location for saving checkpoint was provided
    if args.save_dir:
        save_model(args, model, optimizer, trainloader.dataset.class_to_idx)
        
    
if __name__=="__main__":
    main()