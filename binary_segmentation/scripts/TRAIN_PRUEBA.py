import os
import sys
import math
import torch
from torch.utils.data import DataLoader

from NnUtils import Trainer
from NnUtils import Datasets

# -- IMPORT CUSTOM PATHS: ENABLE EXECUTE FROM TERMINAL  ------------------------------ #

# IMPORTS PATH TO THE PROJECT
current_model_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pycharm_projects_path = os.path.dirname(os.path.dirname(current_model_path))

# IMPORTS PATH TO OTHER PYCHARM PROJECTS
sys.path.append(current_model_path)
sys.path.append(pycharm_projects_path)



class Mink34C(Trainer):
    
    import MinkowskiEngine as ME
    from model.minkunet import MinkUNet34C

    def instantiate_dataset(self):

        self.train_abs_path = os.path.join(self.prefix_path(), self.configuration.train.train_dir.__str__())
        self.valid_abs_path = os.path.join(self.prefix_path(), self.configuration.train.valid_dir.__str__())
        
        self.train_dataset = Datasets.MinkDataset(  _mode='train',
                                                    _root_dir= self.train_abs_path,
                                                    _coord_idx= self.configuration.train.coord_idx,
                                                    _feat_idx=self.configuration.train.feat_idx,
                                                    _feat_ones=self.configuration.train.feat_ones, 
                                                    _label_idx=self.configuration.train.label_idx,
                                                    _normalize=self.configuration.train.normalize,
                                                    _binary=self.configuration.train.binary, 
                                                    _add_range=self.configuration.train.add_range,   
                                                    _voxel_size=self.configuration.train.voxel_size)

        # self.train_dataset.dataset_size = 100

        if self.configuration.train.use_valid_data:
            self.valid_dataset= Datasets.MinkDataset(   _mode='train',
                                                        _root_dir=self.valid_abs_path,
                                                        _coord_idx=self.configuration.train.coord_idx,
                                                        _feat_idx=self.configuration.train.feat_idx, 
                                                        _feat_ones=self.configuration.train.feat_ones,
                                                        _label_idx=self.configuration.train.label_idx,
                                                        _normalize=self.configuration.train.normalize,
                                                        _binary=self.configuration.train.binary, 
                                                        _add_range=self.configuration.train.add_range,   
                                                        _voxel_size=self.configuration.train.voxel_size)  
        else:
            train_size = math.floor(len(self.train_dataset) * self.configuration.train.train_split)
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(74))


    def instantiate_dataloader(self):
        # INSTANCE DATALOADERS
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.configuration.train.batch_size, num_workers=10,
                                      shuffle=True, pin_memory=True, drop_last=True, collate_fn=ME.utils.batch_sparse_collate)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.configuration.train.batch_size, num_workers=10,
                                      shuffle=True, pin_memory=True, drop_last=True, collate_fn=ME.utils.batch_sparse_collate)


    def set_model(self):
        add_len = 1 if self.configuration.train.add_range else 0
        feat_len = len(self.configuration.train.feat_idx) + add_len
        feat_len = 1

        self.model = MinkUNet34C(   in_channels= feat_len, 
                                    out_channels= self.configuration.train.output_classes,
                                    D= len(self.configuration.train.coord_idx)).to(self.device)

        # pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print('-' * 50)
        # print(self.model.__class__.__name__)
        # print(f"N Parameters: {pytorch_total_params}\n\n")


       
    def train(self):
        print('-'*50 + '\n' + 'TRAINING' + '\n' + '-'*50)
        current_clouds = 0
        
        self.model.train()

        for batch, (coords, features, label) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            coords = coords.to(self.device, dtype=torch.float32)
            features = features.to(self.device, dtype=torch.float32)
            label = label.to(self.device, dtype=torch.float32)
           
            in_field = ME.TensorField(  features= features,
                                        coordinates= coords,
                                        quantization_mode= ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                        minkowski_algorithm= ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                                        device= self.device)
            
            # Forward and Evaluate Model
            input = in_field.sparse() #SparseTensorField to SparseTensor
            output = self.model(input)
            pred = output.slice(in_field)
            # pred = self.activation_fn(pred)
            
            prediction = pred.F.squeeze()

            # Calulate loss and backpropagate
            avg_train_loss_ = self.configuration.train.loss_fn(prediction, label)
            avg_train_loss_.backward()
            self.optimizer.step()

            # Print training progress
            current_clouds += features.size(0)
            if batch % 1 == 0 or features.size(0) < self.train_dataloader.batch_size:  # print every (% X) batches
                print(f' - [Batch: {batch + 1 }/{ int(len(self.train_dataloader.dataset) / self.train_dataloader.batch_size)}],'
                    f' / Train Loss: {avg_train_loss_:.4f}')


    def valid(self):
        print('-'*50 + '\n' + 'VALIDATION' + '\n' + '-'*50)
        self.valid_results = Results()
        current_clouds = 0

        self.model.eval()
        with torch.no_grad():
            for batch, (coords, features, label) in enumerate(self.valid_dataloader):

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
                self.valid_results.loss.append(avg_loss.detach().cpu())

                # Compute metrics
                trshld, pred_fix, avg_f1, avg_pre, avg_rec, conf_m = validation_metrics(label, prediction)

                print(f'Threshold: {trshld:.5f}')

                self.valid_results.threshold.append(trshld)
                self.valid_results.f1.append(avg_f1)
                self.valid_results.precision.append(avg_pre)
                self.valid_results.recall.append(avg_rec)
                self.valid_results.conf_matrix.append(conf_m)
                self.valid_results.pred_fix.append(pred_fix)

                current_clouds += features.size(0)

                if batch % 1 == 0 or features.size()[0] < self.valid_dataloader.batch_size:  # print every 10 batches
                    print(f'[Batch: {batch +1 }/{ int(len(self.valid_dataloader.dataset) / self.valid_dataloader.batch_size)}]'
                        f'  [Precision: {avg_pre:.4f}]'
                        f'  [Loss: {avg_loss:.4f}]'
                        f'  [F1: {avg_f1:.4f}]'
                        f'  [Recall: {avg_rec:.4f}]')

        self.compute_mean_valid_results()
        self.add_to_global_results()

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


    def update_learning_rate_custom(self):
        prev_lr = self.get_learning_rate()
        for g in self.optimizer.param_groups:
            g['lr'] = ( prev_lr / 10)

        # self.scheduler.step()

    def update_learning_rate(self):
        if self.configuration.train.lr_scheduler == "step":
            self.lr_scheduler.step()

        elif self.configuration.train.lr_scheduler == "plateau":
            self.lr_scheduler.step(self.valid_avg_loss)


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


