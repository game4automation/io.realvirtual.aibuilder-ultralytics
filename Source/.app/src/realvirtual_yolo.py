
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import glob
import random
import json
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO
import pandas as pd
import seaborn as sns
import shutil
import platform
import psutil

class pp:

    def integrate_data(folder, train_ratio=0.8):
        """
        Organize .bmp files from the source folder into train and val directories.

        :param source_folder: Directory containing the .bmp files.
        :param dest_folder: Destination directory to store the organized files.
        :param train_ratio: Ratio of files to be used for training (default 0.8).
        """
        # Create the destination directories if they don't exist
        train_dir = folder + '/images/train'
        val_dir = folder + '/images/val'
        
        train_labels_dir = folder + '/labels/train'
        val_labels_dir = folder + '/labels/val'

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)

        # Get all .bmp files in the source folder
        bmp_files = list(Path(folder).glob('*.bmp'))

        # Shuffle the files randomly
        random.shuffle(bmp_files)

        # Calculate split index
        split_idx = int(len(bmp_files) * train_ratio)

        # Split into train and val sets
        train_files = bmp_files[:split_idx]
        val_files = bmp_files[split_idx:]

        # Move files to the respective directories
        for file in train_files:
            shutil.move(str(file), str(train_dir) +'/'+ file.name)
            shutil.move(str(file).replace('.bmp', '.txt'), train_labels_dir +'/'+ file.name.replace('.bmp', '.txt'))

        for file in val_files:
            shutil.move(str(file), str(val_dir) +'/'+ file.name)
            shutil.move(str(file).replace('.bmp', '.txt'), val_labels_dir +'/'+ file.name.replace('.bmp', '.txt'))
        

        print(f"Moved {len(train_files)} files to {train_dir}")
        print(f"Moved {len(val_files)} files to {val_dir}")


    def create_dataset_yaml(folder, train_ratio=0.7, val_ratio=0.2):
        """
        Creates a dataset YAML file for object detection.

        Args:
            folder: The path to the dataset folder.
            train_ratio: The proportion of files to use for training (default: 0.7).
            val_ratio: The proportion of files to use for validation (default: 0.2).
            test_ratio: The proportion of files to use for testing (default: 0.1).
        """

        label_file = f"{folder}/labels.json"
        labels = []
        with open(label_file) as f:
            labels = json.load(f)['labels']

        class_names = [label["name"] for label in labels]

        print(f"Classes: {class_names}")
        

        # Create the YAML file
        with open(f"{folder}/dataset.yaml", "w") as f:
            f.write("path: " + folder + "\n")
            #f.write("path: .\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f'nc: {len(class_names)}\n')
            f.write(f"names: {class_names}\n")

    
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


    def prepare_dataset(folder):
        print('Preparing dataset')
        yolo.integrate_data(folder)
        yolo.create_dataset_yaml(folder)


    def load_yaml_config(file):
        with open(file) as f:
            return yaml.safe_load(f)

    def load_json_config(file):
        with open(file) as f:
            return json.load(f)

    def get_sample_files(dataset, mode):
        # datasetin a yaml in coco format
        d = pp.load_yaml_config(dataset)
        samples = glob.glob(f'{d["path"]}/{d[mode]}/*')
        return samples
    
    def load_yaml(file):
        with open(file) as f:
            return yaml.safe_load(f)

    
    def load_samples(folder):
        # Load samples from folder

        # annotation format is the yolorformat for obbs
        # {labelId} {center.x} {center.y} {size.x} {size.y}

        samples = []
        annotations = []

        files = glob.glob(f'{folder}/images/train/*.bmp')

        for file in files:
            # read image as rgb
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            # convert to bgr
            #img = img[..., [2, 1, 0]]
            samples.append(img)

            #samples.append(cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED))
            ann = np.genfromtxt(os.path.join(folder, file.replace('.bmp', '.txt').replace('/images/', '/labels/')), delimiter=' ')
            annotations.append(ann)

        print('Loaded', len(samples), 'samples')
        print('Sample shape:', samples[0].shape)
        print('Annotation shape:', annotations[0].shape)


        return samples, annotations


    def show_sample(samples, annotations, i):
        
        img = samples[i]
        img = img[..., [2, 1, 0]]

        plt.imshow(img)
        plt.axis('off')

        plt.title('Sample ' + str(i))

        ry, rx = samples[i].shape[:2]
        # foreach row in annotations[i] draw the obb
        for row in annotations[i]:
            label, cx, cy, w, h = row

            cx *= rx
            cy *= ry
            w *= rx
            h *= ry

            print(row)
            
            # use colors from matplotlib
            color = plt.cm.Spectral((label % 10)/10.0)

            plt.plot([cx - w/2, cx + w/2, cx + w/2, cx - w/2, cx - w/2],
                    [cy - h/2, cy - h/2, cy + h/2, cy + h/2, cy - h/2],
                    color=color)

        plt.show()


    def show_sample_obb(samples, annotations, i):
        
        img = samples[i]
        img = img[..., [2, 1, 0]]

        print(img.shape)

        plt.imshow(img)
        plt.axis('off')

        plt.title('Sample ' + str(i))

        ry, rx = samples[i].shape[:2]
        # foreach row in annotations[i] draw the obb
        for row in annotations[i]:
            label, x1, y1, x2, y2, x3, y3, x4, y4 = row

            x1 *= rx
            x2 *= rx
            x3 *= rx
            x4 *= rx

            y1 *= ry
            y2 *= ry
            y3 *= ry
            y4 *= ry

            print(row)
            
            # use colors from matplotlib
            color = plt.cm.Spectral((label % 10)/10.0)

            plt.plot([x1,x2,x3,x4],
                    [y1,y2,y3,y4],
                    color=color)

        plt.show()

    
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log(s):
    print(s)

def train(config_path):

    run_folder = './run'

    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    
    config = pp.load_json_config(config_path)


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

    statusfile = f'{run_folder}/{model_name}/status.txt'

    old_run = f'{run_folder}/{model_name}'
    
    device = pp.get_device()

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
        imgsz = imgsz
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

    # delete all weights with epoch in the name
    for file in glob.glob(f'{run_folder}/{model_name}/weights/epoch*.pt'):
        os.remove(file)

    


def create_summary_plots(folder, model_name):
    
  

    summary_folder = f'{folder}/{model_name}/summary'
    os.makedirs(summary_folder, exist_ok=True)

    # get csv at run/trainig/results.csv
    df = pd.read_csv(f'{folder}/{model_name}/results.csv')

    # strip the whitespaces from the column names
    df.columns = df.columns.str.strip()

    # set high dpi for better quality
    plt.rcParams['figure.dpi'] = 150

    # create custom darkmode style
    plt.style.use('dark_background')


    style = sns.axes_style()
    style['figure.facecolor'] = 'grey'
    style['axes.facecolor'] = 'grey'
    style['axes.edgecolor'] = 'white'
    style['axes.labelcolor'] = 'white'
    style['xtick.color'] = 'white'
    style['ytick.color'] = 'white'
    style['grid.color'] = 'white'
    style['grid.linestyle'] = '--'

    sns.set(style=style)


    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


    sns.lineplot(x=df['epoch'], y=df['train/box_loss'], data=df, label='Train', color=colors[0])
    sns.lineplot(x=df['epoch'],  y=df['val/box_loss'], data=df, label='Validation', color=colors[1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Box Loss')
    plt.margins(0)
    plt.tight_layout()
    plt.savefig(f'{summary_folder}/box_loss.png')
    plt.clf()





    sns.lineplot(x=df['epoch'], y=df['train/cls_loss'], data=df, label='Train', color=colors[0])
    sns.lineplot(x=df['epoch'], y=df['val/cls_loss'], data=df, label='Validation', color=colors[1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Class Loss')
    plt.margins(0)
    plt.tight_layout()
    plt.savefig(f'{summary_folder}/class_loss.png')
    plt.clf()



    metrics = []
    for col in df.columns:
        if 'metrics' in col:
            metrics.append(col.removeprefix('metrics/'))

    for metric in metrics:
        sns.lineplot(x=df['epoch'], y=df[f'metrics/{metric}'], data=df, color=colors[0])
        plt.xlabel('Epoch')
        plt.ylabel(metric.lower())
        plt.title(metric)
        plt.margins(0)
        plt.tight_layout()
        plt.savefig(f'{summary_folder}/{metric}.png')
        plt.clf()



def show_results(results, model):

        for r in results:
            r = r.to('cpu').numpy()

            img = r.orig_img.copy()
            # switch bgr to rgb
            img = img[..., [2, 1, 0]]

            plt.imshow(img)
            plt.axis('off')

            

            labels = []
            for i,box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = box
                cls = r.boxes.cls[i]

                color = plt.cm.Set3(int(cls) % 12)

                label = model.names[int(cls)]

                if label not in labels:
                    labels.append(label)
                    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, label=label)
                else:
                    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color)
                
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.show()

