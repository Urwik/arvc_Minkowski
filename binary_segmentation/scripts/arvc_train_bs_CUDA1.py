import numpy as np
import os
import shutil
import math
from torch.utils.data import DataLoader
import torch
import socket
import sys
import sklearn.metrics as metrics
import yaml
from datetime import datetime
from tqdm import tqdm
import warnings
import open3d as o3d
import MinkowskiEngine as ME
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
from model.minkunet import MinkUNet34C


def train(device_, train_loader_, model_, loss_fn_, optimizer_):
    loss_lst = []
    current_clouds = 0
    current_loss = 0

    # TRAINING
    print('-' * 50)
    print('TRAINING')
    print('-'*50)
    model_.train()
    for batch, (coords, features, label) in enumerate(train_loader_):
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

        avg_train_loss_ = loss_fn_(prediction, label)
        loss_lst.append(avg_train_loss_.item())

        optimizer_.zero_grad()
        avg_train_loss_.backward()
        optimizer_.step()

        current_clouds += features.size(0)

        if batch % 10 == 0 or features.size(0) < train_loader_.batch_size:  # print every (% X) batches
            print(f' - [Batch: {batch}/{len(train_loader_)}],'
                  f' / Train Loss: {avg_train_loss_:.4f}')

    return loss_lst


def valid(device_, dataloader_, model_, loss_fn_):

    # VALIDATION
    print('-' * 50)
    print('VALIDATION')
    print('-'*50)
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

            trshld, pred_fix, avg_f1, avg_pre, avg_rec, conf_m = compute_metrics(label, prediction)
            trshld_lst.append(trshld)
            f1_lst.append(avg_f1)
            pre_lst.append(avg_pre)
            rec_lst.append(avg_rec)
            conf_m_lst.append(conf_m)

            current_clouds += features.size(0)

            if batch % 10 == 0 or features.size()[0] < dataloader_.batch_size:  # print every 10 batches
                print(f'  [Batch: {batch}/{len(dataloader_)}],'
                      f'  [Loss: {avg_loss:.4f}]'
                      f'  [Precision: {avg_pre:.4f}]'
                      f'  [Recall: {avg_rec:.4f}'
                      f'  [F1 score: {avg_f1:.4f}]')

    return loss_lst, f1_lst, pre_lst, rec_lst, conf_m_lst, trshld_lst


def compute_metrics(label_, pred_):

    pred = pred_.cpu().numpy()
    label = label_.cpu().numpy().astype(int)
    trshld = compute_best_threshold(pred, label)
    pred = np.where(pred > trshld, 1, 0).astype(int)

    f1_score = metrics.f1_score(label, pred, average='binary')
    precision_ = metrics.precision_score(label, pred)
    recall_ = metrics.recall_score(label, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(label, pred, labels=[0,1]).ravel()


    return trshld, pred, f1_score, precision_, recall_, (tn, fp, fn, tp )


def compute_best_threshold(pred_, gt_):
    trshld_per_cloud = []
    method_ = THRESHOLD_METHOD

    if method_ == "roc":
        fpr, tpr, thresholds = metrics.roc_curve(gt_, pred_)
        gmeans = np.sqrt(tpr * (1 - fpr))
        index = np.argmax(gmeans)
        trshld_per_cloud.append(thresholds[index])

    elif method_ == "pr":
        precision_, recall_, thresholds = metrics.precision_recall_curve(gt_, pred_)
        f1_score_ = (2 * precision_ * recall_) / (precision_ + recall_)
        index = np.argmax(f1_score_)
        trshld_per_cloud.append(thresholds[index])

    elif method_ == "tuning":
        thresholds = np.arange(0.0, 1.0, 0.0001)
        f1_score_ = np.zeros(shape=(len(thresholds)))
        for index, elem in enumerate(thresholds):
            prediction_ = np.where(pred_ > elem, 1, 0).astype(int)
            f1_score_[index] = metrics.f1_score(gt_, prediction_)

        index = np.argmax(f1_score_)
        trshld_per_cloud.append(thresholds[index])
    else:
        print('Error in the name of the method to use for compute best threshold')

    return sum(trshld_per_cloud)/len(trshld_per_cloud)


if __name__ == '__main__':

    # Files = os.listdir(os.path.join(current_model_path, 'config'))
    Files = ['config_1.yaml']

    for configFile in Files:
        start_time = datetime.now()

        # --------------------------------------------------------------------------------------------#
        # GET CONFIGURATION PARAMETERS
        CONFIG_FILE = configFile
        config_file_abs_path = os.path.join(current_model_path, 'config', CONFIG_FILE)
        with open(config_file_abs_path) as file:
            config = yaml.safe_load(file)

        TRAIN_DIR= config["train"]["TRAIN_DIR"]
        VALID_DIR= config["train"]["VALID_DIR"]
        USE_VALID_DATA= config["train"]["USE_VALID_DATA"]
        VOXEL_SIZE = config["train"]["VOXEL_SIZE"]
        OUTPUT_DIR= config["train"]["OUTPUT_DIR"]
        TRAIN_SPLIT= config["train"]["TRAIN_SPLIT"]
        FEATURES= config["train"]["FEATURES"]
        LABELS= config["train"]["LABELS"]
        NORMALIZE= config["train"]["NORMALIZE"]
        BINARY= config["train"]["BINARY"]
        # DEVICE= config["train"]["DEVICE"]
        DEVICE= 'cuda:1'
        BATCH_SIZE= config["train"]["BATCH_SIZE"]
        EPOCHS= config["train"]["EPOCHS"]
        LR= config["train"]["LR"]
        OUTPUT_CLASSES= config["train"]["OUTPUT_CLASSES"]
        THRESHOLD_METHOD= config["train"]["THRESHOLD_METHOD"]
        TERMINATION_CRITERIA= config["train"]["TERMINATION_CRITERIA"]
        EPOCH_TIMEOUT= config["train"]["EPOCH_TIMEOUT"]

        # --------------------------------------------------------------------------------------------#
        # CHANGE PATH DEPENDING ON MACHINE
        machine_name = socket.gethostname()
        if machine_name == 'arvc-Desktop':
            TRAIN_DATA = os.path.join('/media/arvc/data/datasets', TRAIN_DIR)
            VALID_DATA = os.path.join('/media/arvc/data/datasets', VALID_DIR)
        else:
            TRAIN_DATA = os.path.join('/home/arvc/Fran/data/datasets', TRAIN_DIR)
            VALID_DATA = os.path.join('/home/arvc/Fran/data/datasets', VALID_DIR)
        # --------------------------------------------------------------------------------------------#
        # CREATE A FOLDER TO SAVE TRAINING
        OUT_DIR = os.path.join(current_model_path, OUTPUT_DIR)
        folder_name = datetime.today().strftime('%y%m%d%H%M%S')
        OUT_DIR = os.path.join(OUT_DIR, folder_name)
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)

        shutil.copyfile(config_file_abs_path, os.path.join(OUT_DIR, 'config.yaml'))

        
        # SELECT DEVICE
        if torch.cuda.is_available():
            device = torch.device(DEVICE)
        else:
            device = torch.device("cpu")

        torch.cuda.set_device(device=device)
        # ------------------------------------------------------------------------------------------------------------ #
        # INSTANCE DATASET
        train_dataset = minkDataset(root_dir=TRAIN_DATA,
                                   features=FEATURES,
                                   labels=LABELS,
                                   normalize=NORMALIZE,
                                   binary=BINARY,
                                   voxel_size_=VOXEL_SIZE)

        if USE_VALID_DATA:
            valid_dataset = minkDataset(root_dir=VALID_DATA,
                                       features=FEATURES,
                                       labels=LABELS,
                                       normalize=NORMALIZE,
                                       binary=BINARY,
                                       voxel_size_=VOXEL_SIZE)
        else:
            # SPLIT VALIDATION AND TRAIN
            train_size = math.floor(len(train_dataset) * TRAIN_SPLIT)
            val_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size],
                                                                         generator=torch.Generator().manual_seed(74))


        # INSTANCE DATALOADERS
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE, 
            num_workers=10,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=ME.utils.batch_sparse_collate)

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            num_workers=10,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=ME.utils.batch_sparse_collate)

        # ------------------------------------------------------------------------------------------------------------ #


        model = MinkUNet34C(in_channels=len(FEATURES), out_channels=OUTPUT_CLASSES, D=3).to(device)
        
        loss_fn = torch.nn.BCELoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        # optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # ------------------------------------------------------------------------------------------------------------ #
        # --- TRAIN LOOP --------------------------------------------------------------------------------------------- #
        print('TRAINING ON: ', device)
        epoch_timeout_count = 0

        if TERMINATION_CRITERIA == "loss":
            best_val = 1
        else:
            best_val = 0

        f1, precision, recall, conf_matrix, train_loss, valid_loss, threshold = [], [], [], [], [], [], []

        for epoch in range(EPOCHS):
            print(f"EPOCH: {epoch} {'-' * 50}")
            epoch_start_time = datetime.now()

            train_results = train(device_=device,
                                  train_loader_=train_dataloader,
                                  model_=model,
                                  loss_fn_=loss_fn,
                                  optimizer_=optimizer)

            valid_results = valid(device_=device,
                                  dataloader_=valid_dataloader,
                                  model_=model,
                                  loss_fn_=loss_fn)

            # GET RESULTS
            train_loss.append(train_results)
            valid_loss.append(valid_results[0])
            f1.append(valid_results[1])
            precision.append(valid_results[2])
            recall.append(valid_results[3])
            conf_matrix.append(valid_results[4])
            threshold.append(valid_results[5])

            print('-' * 50)
            print('DURATION:')
            print('-' * 50)
            epoch_end_time = datetime.now()
            print('Epoch Duration: {}'.format(epoch_end_time-epoch_start_time))

            # SAVE MODEL AND TEMINATION CRITERIA
            if TERMINATION_CRITERIA == "loss":
                last_val = np.mean(valid_results[0])
                if last_val < best_val:
                    torch.save(model.state_dict(), OUT_DIR + f'/best_model.pth')
                    best_val = last_val
                    epoch_timeout_count = 0
                elif epoch_timeout_count < EPOCH_TIMEOUT:
                    epoch_timeout_count += 1
                else:
                    break
            elif TERMINATION_CRITERIA == "precision":
                last_val = np.mean(valid_results[2])
                if last_val > best_val:
                    torch.save(model.state_dict(), OUT_DIR + f'/best_model.pth')
                    best_val = last_val
                    epoch_timeout_count = 0
                elif epoch_timeout_count < EPOCH_TIMEOUT:
                    epoch_timeout_count += 1
                else:
                    break
            elif TERMINATION_CRITERIA == "f1_score":
                last_val = np.mean(valid_results[1])
                if last_val > best_val:
                    torch.save(model.state_dict(), OUT_DIR + f'/best_model.pth')
                    best_val = last_val
                    epoch_timeout_count = 0
                elif epoch_timeout_count < EPOCH_TIMEOUT:
                    epoch_timeout_count += 1
                else:
                    break
            else:
                print("WRONG TERMINATION CRITERIA")
                exit()


        # SAVE RESULTS
        np.save(OUT_DIR + f'/train_loss', np.array(train_loss))
        np.save(OUT_DIR + f'/valid_loss', np.array(valid_loss))
        np.save(OUT_DIR + f'/f1_score', np.array(f1))
        np.save(OUT_DIR + f'/precision', np.array(precision))
        np.save(OUT_DIR + f'/recall', np.array(recall))
        np.save(OUT_DIR + f'/conf_matrix', np.array(conf_matrix))
        np.save(OUT_DIR + f'/threshold', np.array(threshold))

        end_time = datetime.now()
        print('Total Training Duration: {}'.format(end_time-start_time))
        print("Training Done!")
