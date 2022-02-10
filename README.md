# TrajGAT

This is a graph-based approach, which focusses on modeling the hierarchical spatial structure and improving the performance of long trajectory similarity computation.

## Require Packages
Pytorch, Numpy, Yaml, Dgl, Networkx, Pickle, Scipy, Tensorboard, Tqdm, trajectory_distance

## Running Procedures

### Create Folders
Please create 2 empty folders:

* `data`: Path of the original data which is organized to a trajectory list. Each trajectory in it is a list of coordinate tuples (lon, lat).

* `model/wts`: It is used for placing the best TrajGAT model parameters of training.

### Download Data
Due to the file size limit, we put the dataset on other sites. Please first download the data and put it in `data` folder. The long trajectory dataset of Porto can be download at:  https://drive.google.com/drive/folders/1hORrqGXXPZWiQXKVzAj0EFU6CYgIgeHd?usp=sharing

### Training & Evaluating
To train TrajGAT model, run the following command:
```bash
python main.py --config=model_config.yaml --gpu=0
```
It trains TrajGAT under the supervision of metric distance. The parameters of TrajGAT can be modified in `model_config.yaml`.