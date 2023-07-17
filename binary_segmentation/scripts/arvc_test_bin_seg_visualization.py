import numpy as np
import pandas as pd
import os
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


from arvc_Utils.Datasets import vis_minkDataset
from arvc_Utils.pointcloudUtils import np2ply
from model.minkunet import MinkUNet34C



def test(device_, dataloader_, model_, loss_fn_):
    # TEST
    model_.eval()
    f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst, trshld_lst = [], [], [], [], [], []
    current_clouds = 0

    with torch.no_grad():
        for batch, (coords, features, label) in enumerate(dataloader_):

            coords = coords.to(device_, dtype=torch.float32)
            features = features.to(device_, dtype=torch.float32)
            label = label.to(device_, dtype=torch.float32)


            in_field = ME.TensorField(
                features= features,
                coordinates= (coords / VOXEL_SIZE),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=device_)

            # Forward
            input = in_field.sparse()
            output = model_(input)
            pred = output.slice(in_field)

            prediction = pred.F.squeeze()
            avg_loss = loss_fn_(prediction, label)
            loss_lst.append(avg_loss.item())

            pred_fix, avg_f1, avg_pre, avg_rec, conf_m = compute_metrics(label, prediction)
            f1_lst.append(avg_f1)
            pre_lst.append(avg_pre)
            rec_lst.append(avg_rec)
            conf_m_lst.append(conf_m)

            if SAVE_PRED_CLOUDS:
                print(f'Filename: {FILES_LIST[batch][-9:-4]}')
                save_pred_as_ply(coords[:, 1:], pred_fix, PRED_CLOUDS_DIR, FILES_LIST[batch][-9:-4])

            # current_clouds += features.size(0)

            # if batch % 10 == 0 or features.size()[0] < dataloader_.batch_size:  # print every 10 batches
            #     print(f'  [Batch: {batch}/{len(dataloader_)}]'
            #           f'  [Loss: {avg_loss:.4f}]'
            #           f'  [Precision: {avg_pre:.4f}]'
            #           f'  [Recall: {avg_rec:.4f}'
            #           f'  [F1 score: {avg_f1:.4f}]')

    return loss_lst, f1_lst, pre_lst, rec_lst, conf_m_lst



def compute_metrics(label_, pred_):

    pred = pred_.cpu().numpy()
    label = label_.cpu().numpy().astype(int)
    pred = np.where(pred > THRESHOLD, 1, 0).astype(int)

    f1_score = metrics.f1_score(label, pred, average='binary')
    precision_ = metrics.precision_score(label, pred)
    recall_ = metrics.recall_score(label, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(label, pred, labels=[0,1]).ravel()


    return pred, f1_score, precision_, recall_, (tn, fp, fn, tp )


def save_pred_as_ply(coords_, pred_fix_, out_dir_, filename_):

    feat_xyzlabel = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u4')]
    
    coords_ = coords_.detach().cpu().numpy()
    pred_fix_ = pred_fix_[:,None]

    cloud = np.hstack((coords_, pred_fix_))
    print(f'Out_dir: {out_dir_}')
    print(f'Filename: {filename_}')


    np2ply(cloud, out_dir_, filename_, features=feat_xyzlabel, binary=True)


def get_extend_clouds():
    csv_file = os.path.join(MODEL_PATH, 'representative_clouds.csv')
    df = pd.read_csv(csv_file)
    extended_clouds_ = df.iloc[0].tolist()
    extended_clouds_ = np.unique(extended_clouds_).tolist()

    return extended_clouds_


if __name__ == '__main__':
    start_time = datetime.now()

    # --------------------------------------------------------------------------------------------#
    # REMOVE MODELS THAT ARE ALREADY EXPORTED
    models_list = os.listdir(os.path.join(current_model_path, 'saved_models'))

    # models_list = ['bs_xyz_bce_vt_loss']

    for MODEL_DIR in models_list:
        print(f'Testing Model: {MODEL_DIR}')

        MODEL_PATH = os.path.join(current_model_path, 'saved_models', MODEL_DIR)

        config_file_abs_path = os.path.join(MODEL_PATH, 'config.yaml')
        with open(config_file_abs_path) as file:
            config = yaml.safe_load(file)

        # DATASET
        TEST_DIR= "ARVCTRUSS/test/ply_xyzlabelnormal"
        FEATURES= config["train"]["FEATURES"]
        VOXEL_SIZE = config["train"]["VOXEL_SIZE"]
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
            VIS_DATA = os.path.join('/media/arvc/data/datasets', 'ARVCTRUSS/test_visualization/ply_xyzlabelnormal')

        else:
            TEST_DATA = os.path.join('/home/arvc/Fran/data/datasets', TEST_DIR)
            VIS_DATA = os.path.join('/home/arvc/Fran/data/datasets', 'ARVCTRUSS/test_visualization/ply_xyzlabelnormal')

        
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
        dataset = vis_minkDataset(
            root_dir = TEST_DATA,
            common_clouds_dir = VIS_DATA,
            extend_clouds = [],
            features= FEATURES,
            labels = LABELS,
            normalize = NORMALIZE,
            binary = BINARY,
            compute_weights=False)

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
            PRED_CLOUDS_DIR = os.path.join(MODEL_PATH, "vis_clouds")
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

        f1_score = np.array(results[1])
        precision = np.array(results[2])
        recall = np.array(results[3])
        confusion_matrix_list = np.array(results[4])
        mean_cf = np.mean(confusion_matrix_list, axis=0)
        median_cf = np.median(confusion_matrix_list, axis=0)
        mean_tp, mean_fp, mean_tn, mean_fn = mean_cf[3], mean_cf[1], mean_cf[0], mean_cf[2]
        med_tp, med_fp, med_tn, med_fn = median_cf[3], median_cf[1], median_cf[0], median_cf[2]
        # files_list = results[5]


        print('\n\n')
        print(f'Threshold: {THRESHOLD}')
        print(f'[Mean F1_score:  {np.mean(f1_score)}] [Median F1_score:  {np.median(f1_score)}]')
        print(f'[Mean Precision: {np.mean(precision)}] [Median Precision: {np.median(precision)}]')
        print(f'[Mean Recall:    {np.mean(recall)}] [Median Recall:    {np.median(recall)}]')
        print(f'[Mean TP: {mean_tp}, FP: {mean_fp}, TN: {mean_tn}, FN: {mean_fn}] '
              f'[Median TP: {med_tp}, FP: {med_fp}, TN: {med_tn}, FN: {med_fn}]')
        print("Done!")