import os, sys, random
"""
Functions for working with filepaths and directories
"""

def check_path(path_to_check):
    """ Check that path exists
        If path exists, return path and True
        Otherwise return path and False
    """
    exists = True
    
    # if slash at end of path, remove
    if path_to_check[-1] == '/':
        path_to_check = path_to_check[:-1]
        
    split_path = path_to_check.split('/')
    filename = split_path.pop(-1)
    
    
    try:
        if len(split_path) == 0:
            if filename not in os.listdir():
                exists = False
            else:
                path_to_check = filename
        elif filename not in os.listdir('/'.join(split_path)):
            exists = False
    except FileNotFoundError:
        exists = False
        
    return path_to_check, exists


def make_dir(dir_to_make):
    os.makedirs(dir_to_make)
    return


def quit(msg):
    sys.exit(msg)
    return

def rand_img_path(img_dir):
    ''' Gets a random image from the test image directory,
        returns the image path
    '''
    # check that image directory exists and remove slash if needed
    img_dir, exists = check_path(img_dir)
    
    # if image directory does not exist, exit app
    if not exists:
        quit("{} does not exist".format(img_dir))
        
    # get test folder for images
    img_dir_test = img_dir + '/test' 
        
    # get random folder in test folder
    rand_test_dir = random.choice(os.listdir(img_dir_test))
    # get random image from random folder in test folder
    img_path = random.choice(os.listdir(img_dir_test + '/' + rand_test_dir))

    return "{}/{}/{}".format(img_dir_test, rand_test_dir, img_path)