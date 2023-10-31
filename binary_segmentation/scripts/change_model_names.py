import os
import sys
import numpy as np


if __name__ == '__main__':
    current_file_path = os.path.realpath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))
    models_directory = os.path.join(parent_directory, 'saved_models')

    for model in os.listdir(models_directory):
        model_path = os.path.join(models_directory, model)
        
        if os.path.isdir(model_path):
            precision = np.load(model_path + '/precision.npy')
            recall = np.load(model_path + '/recall.npy')
            f1_score = np.load(model_path + '/f1_score.npy')

            last_precision = np.mean(precision[-1])
            last_recall = np.mean(recall[-1])
            last_f1 = np.mean(f1_score[-1])

            last_precision = "{:.4f}".format(last_precision)
            last_recall = "{:.4f}".format(last_recall)
            last_f1 = "{:.4f}".format(last_f1)

            os.rename(model_path, models_directory + '/' + model[:12])