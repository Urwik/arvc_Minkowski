import numpy as np
import pandas as pd
import os
import open3d as o3d
import csv
from torch.utils.data import DataLoader
import torch
import sklearn.metrics as metrics
import sys
import socket
import yaml
from tqdm import tqdm
from datetime import datetime
import MinkowskiEngine as ME
import warnings
warnings.filterwarnings('ignore')

# IMPORTS PATH TO THE PROJECT
current_model_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pycharm_projects_path = os.path.dirname(os.path.dirname(current_model_path))
minkowsky_project_path = os.path.abspath('/home/arvc/Antonio/virtual_environments/trav_analysis/MinkowskiEngine')

# IMPORTS PATH TO OTHER PYCHARM PROJECTS
sys.path.append(current_model_path)
sys.path.append(pycharm_projects_path)
sys.path.append(minkowsky_project_path)


from arvc_Utils.Datasets import minkDataset
from arvc_Utils.pointcloudUtils import np2ply
from model.minkunet import MinkUNet34C



def test(device_, dataloader_, model_, loss_fn_):
    # TEST
    model_.eval()
    f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst, trshld_lst = [], [], [], [], [], []

    with torch.no_grad():
        for batch, (coords, features) in enumerate(dataloader_):

            coords = coords.to(device_, dtype=torch.float32)
            features = features.to(device_, dtype=torch.float32)
            
            in_field = ME.TensorField(
                features= features,
                coordinates= coords,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=device_)

            # Forward
            input = in_field.sparse()
            output = model_(input)
            pred = output.slice(in_field)
            pred_prob = pred.F.squeeze()
            pred_prob = pred_prob.detach().cpu().numpy()
            pred_lbl = np.where(pred_prob >= THRESHOLD, 1, 0).astype(int)

            filename = os.path.basename(FILES_LIST[batch])[:-4]

            if SAVE_PRED_CLOUDS:
                save_pred_as_ply(features[:, 0:3], pred_lbl, PRED_CLOUDS_DIR, filename_=filename)

    return FILES_LIST[batch][-9:-4]


def save_pred_as_ply(coords_, pred_fix_, out_dir_, filename_):

    feat_xyzlabel = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u4')]
    
    coords_ = coords_.detach().cpu().numpy()
    pred_fix_ = pred_fix_[:,None]
    
    cloud = np.hstack((coords_, pred_fix_))
    np2ply(cloud, out_dir_, filename_, features=feat_xyzlabel, binary=True)


if __name__ == '__main__':
    start_time = datetime.now()

    # --------------------------------------------------------------------------------------------#
    # REMOVE MODELS THAT ARE ALREADY EXPORTED
    models_list = os.listdir(os.path.join(current_model_path, 'saved_models'))

    # models_list = ['230308141518']

    for MODEL_DIR in models_list:
        print(f'Testing Model: {MODEL_DIR}')

        MODEL_PATH = os.path.join(current_model_path, 'saved_models', MODEL_DIR)

        config_file_abs_path = os.path.join(MODEL_PATH, 'config.yaml')
        with open(config_file_abs_path) as file:
            config = yaml.safe_load(file)

        # DATASET
        TEST_DIR= "ARVCOUSTER/ply_xyznormal"
        FEATURES= config["train"]["FEATURES"]
        if FEATURES == [0,1,2,4,5,6]:
            FEATURES = [0,1,2,3,4,5]
        elif FEATURES == [0,1,2,7]:
            FEATURES = [0,1,2,6]

        VOXEL_SIZE = 0.05 #config["train"]["VOXEL_SIZE"]
        LABELS= config["train"]["LABELS"]
        NORMALIZE= config["train"]["NORMALIZE"]
        BINARY= config["train"]["BINARY"]
        # DEVICE= config["test"]["DEVICE"]
        DEVICE= "cuda:3"
        BATCH_SIZE= 1
        OUTPUT_CLASSES= config["train"]["OUTPUT_CLASSES"]
        SAVE_PRED_CLOUDS= True

        # --------------------------------------------------------------------------------------------#
        # CHANGE PATH DEPENDING ON MACHINE
        machine_name = socket.gethostname()
        if machine_name == 'arvc-Desktop':
            TEST_DATA = os.path.join('/media/arvc/data/datasets', TEST_DIR)

        else:
            TEST_DATA = os.path.join('/home/arvc/Fran/data/datasets', TEST_DIR)

        
        # --------------------------------------------------------------------------------------------#
        # SELECT DEVICE
        if torch.cuda.is_available():
            device = torch.device(DEVICE)
        else:
            device = torch.device("cpu")

        torch.cuda.set_device(device=device)
       
        # --------------------------------------------------------------------------------------------#
        # GET IMPORTANT CLOUDS FROM TOTAL TEST DATASET
        # extended_clouds = get_extend_clouds()

        # --------------------------------------------------------------------------------------------#
        # INSTANCE DATASET
        dataset = minkDataset(
            mode_ = 'test_no_labels',
            root_dir = TEST_DATA,
            features= FEATURES,
            labels = [],
            normalize = NORMALIZE,
            binary = BINARY,
            voxel_size_=VOXEL_SIZE)

        FILES_LIST = dataset.dataset.copy()

        #---------------------------------------------------------------------------------------------#
        # INSTANCE DATALOADER
        test_dataloader = DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE, 
            num_workers=10,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=ME.utils.batch_sparse_collate)

        #---------------------------------------------------------------------------------------------#
        # SELECT DEVICE TO WORK WITH
        if torch.cuda.is_available():
            device = torch.device(DEVICE)
        else:
            device = torch.device("cpu")

        model = MinkUNet34C(in_channels=len(FEATURES), out_channels=OUTPUT_CLASSES, D=3).to(device)

        loss_fn = torch.nn.BCELoss()

        #---------------------------------------------------------------------------------------------#
        # MAKE DIR WHERE TO SAVE THE CLOUDS
        if SAVE_PRED_CLOUDS:
            PRED_CLOUDS_DIR = os.path.join(MODEL_PATH, "test_ouster_clouds")
            if not os.path.exists(PRED_CLOUDS_DIR):
                os.makedirs(PRED_CLOUDS_DIR)

        #---------------------------------------------------------------------------------------------#
        # LOAD TRAINED MODEL
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'best_model.pth'), map_location=device))
        threshold = np.load(MODEL_PATH + f'/threshold.npy')
        THRESHOLD = np.mean(threshold[-1])

        print('-'*50)
        print('TESTING ON: ', device)
        results = test(device_=device,
                       dataloader_=test_dataloader,
                       model_=model,
                       loss_fn_=loss_fn)

        print("Done!")