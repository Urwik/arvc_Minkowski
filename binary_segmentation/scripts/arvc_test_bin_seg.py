import numpy as np
import pandas as pd
import os
import time
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
    current_clouds = 0

    with torch.no_grad():
        for batch, (coords, features, label) in enumerate(dataloader_):

            coords = coords.to(device_, dtype=torch.float32)
            features = features.to(device_, dtype=torch.float32)
            label = label.to(device_, dtype=torch.float32)

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

            prediction = pred.F.squeeze()
            avg_loss = loss_fn_(prediction, label)
            loss_lst.append(avg_loss.item())

            pred_fix, avg_f1, avg_pre, avg_rec, conf_m = compute_metrics(label, prediction)
            f1_lst.append(avg_f1)
            pre_lst.append(avg_pre)
            rec_lst.append(avg_rec)
            conf_m_lst.append(conf_m)

            if SAVE_PRED_CLOUDS:
                save_pred_as_ply(features[:, 0:3], pred_fix, PRED_CLOUDS_DIR, FILES_LIST[batch][-9:-4])


            current_clouds += features.size(0)

            if batch % 10 == 0 or features.size()[0] < dataloader_.batch_size:  # print every 10 batches
                print(f'  [Batch: {batch}/{len(dataloader_)}]'
                      f'  [Loss: {avg_loss:.4f}]'
                      f'  [Precision: {avg_pre:.4f}]'
                      f'  [Recall: {avg_rec:.4f}'
                      f'  [F1 score: {avg_f1:.4f}]')

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
    np2ply(cloud, out_dir_, filename_, features=feat_xyzlabel, binary=True)


def get_representative_clouds(f1_score_, precision_, recall_, files_list_):

    print('-'*50)
    print("Representative Clouds")

    max_f1 = np.max(f1_score_)
    min_f1 = np.min(f1_score_)
    max_pre = np.max(precision_)
    min_pre = np.min(precision_)
    max_rec = np.max(recall_)
    min_rec = np.min(recall_)

    max_f1_idx = list(f1_score_).index(max_f1.item())
    min_f1_idx = list(f1_score_).index(min_f1.item())
    max_pre_idx = list(precision_).index(max_pre.item())
    min_pre_idx = list(precision_).index(min_pre.item())
    max_rec_idx = list(recall_).index(max_rec.item())
    min_rec_idx = list(recall_).index(min_rec.item())

    clouds = {
        'Max_F1': [files_list_[max_f1_idx]],
        'Min_F1': [files_list_[min_f1_idx]],
        'Max_Pre': [files_list_[max_pre_idx]],
        'Min_Pre': [files_list_[min_pre_idx]],
        'Max_Rec': [files_list_[max_rec_idx]],
        'Min_Rec': [files_list_[min_rec_idx]]
    }

    df = pd.DataFrame(clouds)
    csv_path = os.path.join(MODEL_PATH, 'representative_clouds.csv')
    df.to_csv(csv_path, index=False, sep=',')

    print(f'Max f1 cloud: {files_list_[max_f1_idx]}')
    print(f'Min f1 cloud: {files_list_[min_f1_idx]}')
    print(f'Max precision cloud: {files_list_[max_pre_idx]}')
    print(f'Min precision cloud: {files_list_[min_pre_idx]}')
    print(f'Max recall cloud: {files_list_[max_rec_idx]}')
    print(f'Min recall cloud: {files_list_[min_rec_idx]}')


def get_exported_results():
    csv_file = os.path.join(current_model_path, 'results.csv')
    my_csv = pd.read_csv(csv_file)
    exported_results = list(my_csv['MODEL_NAME'].to_numpy())

    return exported_results


def export_results(f1_score_, precision_, recall_, tp_, fp_, tn_, fn_):
    csv_file = os.path.join(current_model_path, 'results.csv')

    data_list = [MODEL_DIR, FEATURES, VOXEL_SIZE, config["train"]["LOSS"], config["train"]["TERMINATION_CRITERIA"],
                 config["train"]["THRESHOLD_METHOD"], config["train"]["USE_VALID_DATA"],
                 precision_, recall_, f1_score_, tp_, fp_, tn_, fn_]

    with open(csv_file, 'a') as file_object:
        writer = csv.writer(file_object)
        writer.writerow(data_list)
        file_object.close()


if __name__ == '__main__':
    start_time = datetime.now()

    # --------------------------------------------------------------------------------------------#
    # REMOVE MODELS THAT ARE ALREADY TESTED
    # exported_models = get_exported_results()
    # models_list = os.listdir(os.path.join(current_model_path, 'saved_models'))
    # for exported_model in exported_models:
    #     models_list.remove(str(exported_model))

    # REWRITE MODELS TO TEST
    models_list = ['230310174515']

    # FOR EACH MODEL
    for MODEL_DIR in models_list:
        print(f'Testing Model: {MODEL_DIR}')

        MODEL_PATH = os.path.join(current_model_path, 'saved_models', MODEL_DIR)
        config_file_abs_path = os.path.join(MODEL_PATH, 'config.yaml')
        with open(config_file_abs_path) as file:
            config = yaml.safe_load(file)

        # PARSE CONFIG ARGS
        TEST_DIR = 'ARVCTRUSS/unique_cloud' #config["test"]["TEST_DIR"]
        FEATURES = config["train"]["FEATURES"]
        VOXEL_SIZE = config["train"]["VOXEL_SIZE"]
        LABELS = config["train"]["LABELS"]
        NORMALIZE = config["train"]["NORMALIZE"]
        BINARY = config["train"]["BINARY"]
        DEVICE = "cuda:0"    # config["test"]["DEVICE"]
        BATCH_SIZE = 1
        OUTPUT_CLASSES = config["train"]["OUTPUT_CLASSES"]
        SAVE_PRED_CLOUDS = True #config["test"]["SAVE_PRED_CLOUDS"]

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
        # INSTANCE DATASET
        dataset = minkDataset(root_dir = TEST_DATA,
                             features= FEATURES,
                             labels = LABELS,
                             normalize = NORMALIZE,
                             binary = BINARY,
                             voxel_size_=VOXEL_SIZE)

        FILES_LIST = dataset.dataset

        #---------------------------------------------------------------------------------------------#
        # INSTANCE DATALOADER
        test_dataloader = DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE, 
            num_workers=10,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
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
            PRED_CLOUDS_DIR = os.path.join(MODEL_PATH, "Unique_pred_clouds")
            if not os.path.exists(PRED_CLOUDS_DIR):
                os.makedirs(PRED_CLOUDS_DIR)

        #---------------------------------------------------------------------------------------------#
        # LOAD TRAINED MODEL
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'best_model.pth'), map_location=device))
        threshold = np.load(MODEL_PATH + f'/threshold.npy')
        THRESHOLD = np.mean(threshold[-1])

        print('-'*50)
        print('TESTING ON: ', device)
        start_time = time.time()
        results = test(device_=device,
                       dataloader_=test_dataloader,
                       model_=model,
                       loss_fn_=loss_fn)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Model: ', MODEL_DIR, ' Inf. Time: ', elapsed_time)
        f1_score = np.array(results[1])
        precision = np.array(results[2])
        recall = np.array(results[3])
        confusion_matrix_list = np.array(results[4])
        mean_cf = np.mean(confusion_matrix_list, axis=0)
        median_cf = np.median(confusion_matrix_list, axis=0)
        mean_tp, mean_fp, mean_tn, mean_fn = mean_cf[3], mean_cf[1], mean_cf[0], mean_cf[2]
        med_tp, med_fp, med_tn, med_fn = median_cf[3], median_cf[1], median_cf[0], median_cf[2]
        
        get_representative_clouds(f1_score, precision, recall, files_list_=FILES_LIST)
        export_results(np.mean(f1_score), np.mean(precision), np.mean(recall), mean_tp, mean_fp, mean_tn, mean_fn)

        print('\n\n')
        print(f'Threshold: {THRESHOLD}')
        print(f'[Mean F1_score:  {np.mean(f1_score)}] [Median F1_score:  {np.median(f1_score)}]')
        print(f'[Mean Precision: {np.mean(precision)}] [Median Precision: {np.median(precision)}]')
        print(f'[Mean Recall:    {np.mean(recall)}] [Median Recall:    {np.median(recall)}]')
        print(f'[Mean TP: {mean_tp}, FP: {mean_fp}, TN: {mean_tn}, FN: {mean_fn}] '
              f'[Median TP: {med_tp}, FP: {med_fp}, TN: {med_tn}, FN: {med_fn}]')
        print("Done!")