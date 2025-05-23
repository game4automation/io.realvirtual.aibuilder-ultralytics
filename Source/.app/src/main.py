import sys
import os
import torch
import glob
import json
import cv2
from ultralytics import YOLO
import platform
import psutil
import onnx


# for bundling, the default_settings need to be moved to ultralytics data folder in dist under cfg/defualt.yaml

def check_device():
        if torch.cuda.is_available():
            # If CUDA is available, print GPU information
            print("Device          : " + torch.cuda.get_device_name(torch.cuda.current_device()))
            print("PyTorch version : " + torch.__version__)
            print("CUDA version    : " + torch.version.cuda)
            print(f"CUDNN version   : {torch.backends.cudnn.version()}")
        else:
            # If CUDA is not available, print CPU information
            print("CUDA is not available. Running on CPU.")
            print("CPU             : " + platform.processor())
            print("Number of cores : " + str(psutil.cpu_count(logical=True)))
            print("PyTorch version : " + torch.__version__)

def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_json_config(file):
        with open(file) as f:
            return json.load(f)


def train(config_path):

    run_folder = './run'

    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    
    config = load_json_config(config_path)


    print()
    for key in config:
        print(f'{key} : {config[key]}')
    print()
    

    data_folder = config['data_folder']
    export_folder = config['export_folder']
    base_model = config['base_model']
    model_name = config['model_name']
    epochs = config['epochs']
    resume = config['resume']
    imgsz = config['imgsz']
    cache = config['cache']

    statusfile = f'{run_folder}/{model_name}/status.txt'

    old_run = f'{run_folder}/{model_name}'
    
    device = get_device()

    model = None

    if resume or base_model == 'last':
        weight_path = glob.glob(f'{old_run}/weights/last.pt')[0]
        print()
        print(f'Resuming training from {weight_path}')
        print()
        model = YOLO(weight_path)

    else:
        model = YOLO(f"{base_model}.pt")
    
    model.to(device)
    
    print()
    print("Model Device : " + str(model.device))
    print()

    # delete all event files
    for file in glob.glob(f'{run_folder}/{model_name}/events.out.tfevents*'):
        os.remove(file)


    # actual model training
    training_result = model.train(
        data = data_folder,
        epochs = epochs,
        device = 0,
        project = run_folder,  # Specify the root folder to save the run
        name = model_name,  # The name of the run
        exist_ok = True,  # If True, continue training in the same folder if it exists
        plots = False,
        pretrained = True,
        verbose = False,
        resume = resume,
        save_period = 1,
        save = True,
        imgsz = imgsz,
        cache = cache,
        workers = 0
    )

    
    with open(statusfile, 'w') as f:
        f.write('done')


    # export the best model to onnx

    model = YOLO(f"{run_folder}/{model_name}/weights/best.pt")

    export_root = f'{export_folder}/{model_name}'

    if not os.path.exists(export_root):
        os.makedirs(export_root)

    model.export(format="onnx")  

    onnx_path = f'{export_root}/{model_name}.onnx'

    # delete the old onnx file if it exists
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

    # copy onnx to export folder
    os.rename(f'{run_folder}/{model_name}/weights/best.onnx', onnx_path)

    print()
    print("-------------------------------------------------------------------")
    print("ONNX model exported to " + onnx_path)
    print("-------------------------------------------------------------------")
    print()

    # delete all weights with epoch in the name
    for file in glob.glob(f'{run_folder}/{model_name}/weights/epoch*.pt'):
        os.remove(file)

   

if __name__ == '__main__':
    
    args = sys.argv

    if '--multiprocessing-fork' in args:
        pass  # Do nothing if it's a multiprocessing fork

    elif len(args) < 2:
        print("please provide the path to the config file")
        pass

    else:
            # change working directory
            os.makedirs('.results', exist_ok=True)
            os.chdir('.results')

            config_path = args[1]

            print()
            print("------------------------------------------------------------")
            print("--------------- RealVirtual AI Builder 1.0 -----------------")
            print("------------------------------------------------------------")
            print()

            print("Training session for custom object detection model : ")
            print()
            check_device()
            print()

            train(config_path)

        

