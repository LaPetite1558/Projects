# Imports python modules
import torch
from torch import nn
from get_args import get_predict_args
from preprocess import process_image, get_cat_to_name
from model_utils import load_model
from os_utils import rand_img_path, quit
"""
Usage: python predict.py [arguments]
arguments:
    1. img_path - if given, requires a valid path to an image to be specified or 'random'
                  which will prompt the function to pick a random image from the test folder
                  if not given, will use default value 'flowers/test/10/image_07090.jpg'
    2. checkpoint - if given, requires a valid path to a checkpoint to be specified
                    if not given, will use default value 'save/checkpoint.pth'
    3. --top_k - optional, if given, will return specified number of classes and
                 associated probabilities, otherwise, will return default number of
                 5 classes and associated probabilities
    4. --category_names - optional, if given, will use specified file to map classes
                          to category names, otherwise, will default to classes
    5. --actual - optional, if given, will return actual class with predicted
                  classes and probabilities, otherwise will return only predicted
                  classes and probabilities
    6. --gpu - optional, if given, will use GPU if available, if not given, will
               default to CPU
"""

def predict(model, img, top_k):
    with torch.no_grad():
        logx = model.forward(img)
    
    x = torch.exp(logx)
    
    # get top k probabilities and indexes
    top_p, top_idx = x.topk(top_k, dim=1)
    
    # get classes from indexes
    top_class = [k for k,v in model.class_to_idx.items() if v in top_idx.tolist()[0]]
    
    return top_class, top_p.tolist()[0]

def display_results(pred_class, probs, img_path, category_names, show_actual):
    # dictionary to store results
    results = {}
    
    # if file of class names was provided
    # get file and get names
    if category_names:
        cat_to_name = get_cat_to_name(category_names)
        pred_class = [cat_to_name[i] for i in pred_class]
    
    # if argument to show actual class
    # was provided, get actual class
    if show_actual:
        actual = img_path.split('/')[-2]
        # if file of class names was provided
        # get name of actual class
        if category_names:
            actual = cat_to_name[actual]
        results['actual'] = actual
    
    results['class'] = pred_class
    results['probs'] = probs

    # show actual class if available
    if 'actual' in results:
        print("Actual class: {}".format(results['actual']))

    # display classes and probabilities
    print("Class" + (" "*25) + "Probability")
    for c, p in zip(results['class'], results['probs']):
        c_len = len(c)
        print(c + (" "*(30-c_len)) + "{:.6f}".format(p))

        
def main():
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        Returns dictionary containing lists of classes and probabilities 
    '''
    args = get_predict_args()
    
#     print(args)
    
    # if user asks for gpu, enable gpu if available
    if args.gpu and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device(0)
    else:
        print("GPU unavailable. Defaulting to CPU...")
        device = torch.device("cpu")
    
    # get model from checkpoint
    model = load_model(args.checkpoint)
    
    # if user asks app to pick random image
    if args.img_path == 'random':
        img_path = rand_img_path(model.image_dir)
    else:
        img_path = args.img_path
    
    # get processed image
    img = process_image(img_path)
    img = img.float().to(device)
    
#     usr_input = input("Model ready for prediction. Proceed? [y|n] ")
#     if usr_input == 'n':
#         quit("Exiting application...")
    
    model.to(device);
    model.eval()
    
    # get predicted classes and probabilities
    pred_class, probs = predict(model, img, args.top_k)
    
    display_results(pred_class, probs, img_path, 
                    args.category_names, args.actual)
        
        
if __name__=="__main__":
    main()