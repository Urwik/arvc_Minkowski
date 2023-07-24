import os
import sys
import shutil
import socket
from datetime import datetime

import math
import numpy as np
import sklearn.metrics as metrics

import torch
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

import yaml

import warnings
warnings.filterwarnings('ignore')


# -- IMPORT CUSTOM PATHS: ENABLE EXECUTE FROM TERMINAL  ------------------------------ #

# IMPORTS PATH TO THE PROJECT
current_model_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pycharm_projects_path = os.path.dirname(os.path.dirname(current_model_path))
minkowsky_project_path = os.path.abspath('/home/arvc/Fran/PycharmProjects/MinkowskiEngine')

# IMPORTS PATH TO OTHER PYCHARM PROJECTS
sys.path.append(current_model_path)
sys.path.append(pycharm_projects_path)
sys.path.append(minkowsky_project_path)

from NnUtils.Datasets import MinkDataset
from NnUtils.Metrics import test_metrics
from NnUtils.Config import Config as cfg
from model.minkunet import MinkUNet34C

from NnUtils.Utils import bcolors

# ----------------------------------------------------------------------------------- #


class Tester():

    def __init__(self, config_obj_, _model_path):
        self.configuration = config_obj_ #type: cfg
        self.model_path= _model_path
        self.output_dir: str
        self.best_value: float
        self.last_value: float
        self.activation_fn = self.configuration.train.activation_fn
        self.device = self.configuration.train.device
        self.test_results = Results()
        self.global_valid_results = Results()

        self.epoch_timeout_count = 0

        self.setup()


    def setup(self):
        # self.make_outputdir()
        self.set_device()
        self.get_threshold()
        self.instantiate_dataset()
        self.instantiate_dataloader()
        self.set_model()


    def make_outputdir(self):
        OUT_DIR = os.path.join(current_model_path, self.configuration.train.output_dir.__str__())

        folder_name = datetime.today().strftime('%y%m%d%H%M%S')
        OUT_DIR = os.path.join(OUT_DIR, folder_name)

        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)

        self.output_dir = OUT_DIR


    def prefix_path(self):

        machine_name = socket.gethostname()
        prefix_path = ''
        if machine_name == 'arvc-fran':
            prefix_path = '/media/arvc/data/datasets'
        else:
            prefix_path = '/home/arvc/Fran/data/datasets'

        return prefix_path


    def instantiate_dataset(self):

        self.test_abs_path = os.path.join(self.prefix_path(), self.configuration.test.test_dir.__str__())
        
        self.test_dataset = MinkDataset(    _mode='test',
                                            _root_dir= self.test_abs_path,
                                            _coord_idx= self.configuration.train.coord_idx,
                                            _feat_idx=self.configuration.train.feat_idx,
                                            _feat_ones=self.configuration.train.feat_ones, 
                                            _label_idx=self.configuration.train.label_idx,
                                            _normalize=self.configuration.train.normalize,
                                            _binary=self.configuration.train.binary, 
                                            _add_range=self.configuration.train.add_range,   
                                            _voxel_size=self.configuration.train.voxel_size)


    def instantiate_dataloader(self):
        # INSTANCE DATALOADERS
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.configuration.test.batch_size, num_workers=10,
                                      shuffle=False, pin_memory=True, drop_last=True, collate_fn=ME.utils.batch_sparse_collate)


    def set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device(self.configuration.test.device)
        else:
            self.device = torch.device("cpu")


    def set_model(self):
        
        if not self.configuration.train.feat_ones:
            add_len = 1 if self.configuration.train.add_range else 0
            feat_len = len(self.configuration.train.feat_idx) + add_len
        else:
            feat_len = 1

        self.model = MinkUNet34C(   in_channels= feat_len, 
                                    out_channels= self.configuration.train.output_classes,
                                    D= len(self.configuration.train.coord_idx)).to(self.device)

        self.model.load_state_dict(torch.load(os.path.join(self.model_path, 'best_model.pth'), map_location=self.device))

        # pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print('-' * 50)
        # print(self.model.__class__.__name__)
        # print(f"N Parameters: {pytorch_total_params}\n\n")


    def save_global_results(self):
        # SAVE RESULTS
        np.save(self.output_dir + f'/valid_loss', np.array(self.global_valid_results.loss))
        np.save(self.output_dir + f'/f1_score', np.array(self.global_valid_results.f1))
        np.save(self.output_dir + f'/precision', np.array(self.global_valid_results.precision))
        np.save(self.output_dir + f'/recall', np.array(self.global_valid_results.recall))
        np.save(self.output_dir + f'/conf_matrix', np.array(self.global_valid_results.conf_matrix))
        np.save(self.output_dir + f'/threshold', np.array(self.global_valid_results.threshold))

    

    def add_to_global_results(self):
        self.global_valid_results.loss.append(self.valid_avg_loss)
        self.global_valid_results.f1.append(self.valid_avg_f1)
        self.global_valid_results.precision.append(self.valid_avg_precision)
        self.global_valid_results.recall.append(self.valid_avg_recall)
        self.global_valid_results.conf_matrix.append(self.valid_avg_cm)
        self.global_valid_results.threshold.append(self.valid_avg_threshold)


    def compute_mean_valid_results(self):
        self.valid_avg_loss = float(np.mean(self.test_results.loss))
        self.valid_avg_f1 = float(np.mean(self.test_results.f1))
        self.valid_avg_precision = float(np.mean(self.test_results.precision))
        self.valid_avg_recall = float(np.mean(self.test_results.recall))
        self.valid_avg_threshold = float(np.mean(self.test_results.threshold))
        self.valid_avg_cm = np.mean(self.test_results.conf_matrix, axis=0)



    def print_mean_valid_results(self):
        print(f'VALIDATION RESULTS: \n'
                f'  [Precision: {self.valid_avg_precision:.4f}]'
                f'  [Loss: {self.valid_avg_loss:.4f}]'
                f'  [F1: {self.valid_avg_f1:.4f}]'
                f'  [Recall: {self.valid_avg_recall:.4f}]'
                f'  [Threshold: {self.valid_avg_threshold:.4f}]'
                f'  [Confusion Matrix: \n{self.valid_avg_cm}]')        


    def test(self):
        print('-'*50 + '\n' + 'TESTING' + '\n' + '-'*50)
        self.test_results = Results()

        self.model.eval()
        with torch.no_grad():
            for batch, (coords, features, label) in enumerate(self.test_dataloader):

                coords = coords.to(self.device, dtype=torch.float32)
                features = features.to(self.device, dtype=torch.float32)
                label = label.to(self.device, dtype=torch.float32)

                in_field = ME.TensorField(  features= features,
                                            coordinates= coords,
                                            quantization_mode= ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                            minkowski_algorithm= ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                                            device= self.device)


                # Forward
                input = in_field.sparse()
                output = self.model(input)
                pred = output.slice(in_field)
                # pred = self.activation_fn(pred)

                prediction = pred.F.squeeze()

                # Calulate loss and backpropagate
                avg_loss = self.configuration.train.loss_fn(prediction, label)
                self.test_results.loss.append(avg_loss.detach().cpu())

                # Compute metrics
                trshld, pred_fix, avg_f1, avg_pre, avg_rec, conf_m = test_metrics(label, prediction, self.threshold)

                self.test_results.threshold.append(trshld)
                self.test_results.f1.append(avg_f1)
                self.test_results.precision.append(avg_pre)
                self.test_results.recall.append(avg_rec)
                self.test_results.conf_matrix.append(conf_m)
                self.test_results.pred_fix.append(pred_fix)

                # if batch % 10 == 0 or features.size()[0] < self.test_dataloader.batch_size:  # print every 10 batches
                #     print(f'[Batch: {batch +1 }/{ int(len(self.test_dataloader.dataset) / self.test_dataloader.batch_size)}]'
                #         f'  [Precision: {avg_pre:.4f}]'
                #         f'  [Loss: {avg_loss:.4f}]'
                #         f'  [F1: {avg_f1:.4f}]'
                #         f'  [Recall: {avg_rec:.4f}]')

        self.compute_mean_valid_results()
        self.add_to_global_results()


    def get_threshold(self):
        threshold_file_abs_path = os.path.join(self.model_path, 'threshold.npy')
        threshold_list = np.load(threshold_file_abs_path)

        self.threshold = threshold_list[-1] 

class Results:
    def __init__(self):
        self.f1 = []
        self.precision = []
        self.recall = []
        self.conf_matrix = []
        self.loss = []
        self.threshold = []
        self.pred_fix = []




def get_config(_config_file):
    config_file_abs_path = os.path.join(current_model_path, 'config', _config_file)
    config = cfg(config_file_abs_path)

    return config




if __name__ == '__main__':

    MODELS_DIRS = os.listdir(os.path.join(current_model_path, 'saved_models'))

    for MODEL_DIR in MODELS_DIRS:
        print('-'*50)
        print(f'TESTING MODEL: {MODEL_DIR}')
        MODEL_PATH = os.path.join(current_model_path, 'saved_models', MODEL_DIR)
        model_config = os.path.join(MODEL_PATH, 'config.yaml')

        training_start_time = datetime.now()

        config = get_config(model_config)
        
        if config.test.device.__str__() == 'cuda:0':
            torch.cuda.set_device(0)
        elif config.test.device.__str__() == "cuda:1":
            torch.cuda.set_device(1)
        else:
            print('ERROR: DEVICE NOT FOUND')

        tester = Tester(config, MODEL_PATH)

        print('DEVICE: ', tester.device.__str__())
        epoch_start_time = datetime.now()

        tester.test()
        tester.print_mean_valid_results()

        # tester.save_global_results()
        print('\n'); print('-'*50);print(f'{bcolors.GREEN}TEST DONE!{bcolors.ENDC}') 
        print(f'TOTAL INFERENCE DURATION: {bcolors.ORANGE} {datetime.now()-training_start_time}{bcolors.ENDC}'); print('-'*50); print('\n')
