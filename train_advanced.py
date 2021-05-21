import hdbscan
import sys
import os
import models, pipeline_tools, data_utils, processing
import numpy as np
import pandas as pd
from time import time

VALID_SIZE = 0.2
SEED = 333
COLUMN = 'tip_qrs'
ECG_FILE = 'train_ecg_my.npy'
ANN_FILE = 'train_ann_my.npy'
DOWNSAMPLE = 2
WINDOW = 128
NUM_WORKERS = 6


def process_patient(patient):
    ecgs = list()
    anns = list()

    ann = data_utils.get_ann(patient)
    ann = ann[ann['shum'] == 0]
    ann = ann[['pos_R1', COLUMN]]
    ann['patient'] = patient
    ann['pos_R1'] //= DOWNSAMPLE

    ecg = data_utils.get_ecg(patient)
    ecg = processing.process_ecg(ecg, downsample=DOWNSAMPLE)

    for index, row in ann.iterrows():
        ecg_slice = processing.get_slice(ecg, row['pos_R1'], WINDOW, maxval=20, threshold=10)
        if ecg_slice is not None:
            ecgs.append(ecg_slice)
            anns.append(row)

    if len(anns) > 0:
        anns = pd.DataFrame(anns)
        ecgs = np.stack(ecgs)
    else:
        anns = pd.DataFrame([])
        ecgs = np.zeros((0, 2, WINDOW), dtype='float16')
    return anns, ecgs


def expirement(ecg_file, ann_file, patient):
    sys.path.append('../holter')
    params = dict()
    params['anns_file'] = ann_file
    params['ecgs_file'] = ecg_file
    params['valid_size'] = 0.3
    params['seed'] = 333
    params['embedder_class'] = models.AutoEncoder
    params['clusterer_class'] = hdbscan.HDBSCAN
    params['window'] = 128
    params['patient'] = patient

    embedder_params = dict()
    embedder_params['channels_mult'] = 1
    embedder_params['bottleneck'] = 32
    embedder_params['batch_norm'] = True
    embedder_params['valid_size'] = 0.3
    embedder_params['seed'] = 333
    embedder_params['aug_params'] = {'noise': 1.0, 'zero_chan_p': 0.5, 'zero_chan_noise': 2.0}
    embedder_params['num_workers'] = 0
    embedder_params['batch_size'] = 512
    embedder_params['device'] = 'cuda:0'
    embedder_params['lr'] = 0.003
    embedder_params['num_epochs'] = 5
    embedder_params['log_file'] = 'logs.csv'
    embedder_params['preload'] = 'weights_12.pt'
    embedder_params['weights_file'] = 'weights_12.pt'
    embedder_params['train_net'] = True

    clusterer_params = dict()
    clusterer_params['algorithm'] = 'boruvka_kdtree'
    clusterer_params['min_cluster_size'] = 100
    clusterer_params['cluster_selection_epsilon'] = 0.01
    clusterer_params['min_samples'] = 10

    params['embedder_params'] = embedder_params
    params['clusterer_params'] = clusterer_params
    pipeline_tools.run(params)


def expirement_run(ecg_path, rr2_path, output_path, patient=None,):
    sys.path.append('../holter')
    params = dict()
    params['valid_size'] = 0.3
    params['seed'] = 333
    params['embedder_class'] = models.AutoEncoder
    params['clusterer_class'] = hdbscan.HDBSCAN
    params['window'] = 128
    params['patient'] = patient
    params["ECG_path"] = ecg_path
    params["RR2_path"] = rr2_path
    params["output_path"] = output_path
    embedder_params = dict()
    embedder_params['channels_mult'] = 1
    embedder_params['bottleneck'] = 32
    embedder_params['batch_norm'] = True
    embedder_params['valid_size'] = 0.3
    embedder_params['seed'] = 333
    embedder_params['aug_params'] = {'noise': 1.0, 'zero_chan_p': 0.5, 'zero_chan_noise': 2.0}
    embedder_params['num_workers'] = 0
    embedder_params['batch_size'] = 512
    embedder_params['device'] = 'cuda:0'
    embedder_params['lr'] = 0.003
    embedder_params['num_epochs'] = 5
    embedder_params['log_file'] = 'logs.csv'
    embedder_params['preload'] = 'weights_12.pt'
    embedder_params['weights_file'] = 'weights_12.pt'
    embedder_params['train_net'] = False

    clusterer_params = dict()
    clusterer_params['algorithm'] = 'boruvka_kdtree'
    clusterer_params['min_cluster_size'] = 100
    clusterer_params['cluster_selection_epsilon'] = 0.01
    clusterer_params['min_samples'] = 10

    params['embedder_params'] = embedder_params
    params['clusterer_params'] = clusterer_params
    pipeline_tools.run(params)


def main():
    # path_ecg=f"/large/datasets/holter/ecg/{patient}.ecg"
    # path_rr2= f"/large/datasets/holter/rr2/{patient}.rr2"
    print("Введите путь до ЭКГ: \n")
    path_ecg = input()
    filename, file_extension = os.path.splitext(path_ecg)
    while not os.path.isfile(path_ecg) or file_extension != ".ecg":
        print("Некорректный путь до ЭКГ. Введите ещё раз:")
        path_ecg = input()
        filename, file_extension = os.path.splitext(path_ecg)
        # path_ecg=f"/large/datasets/holter/ecg/5050.ecg"
        # path_rr2 = f"/large/datasets/holter/rr2/5050.rr2"
    print("Введите путь до разметки:\n")
    path_rr2 = input()
    filename, file_extension = os.path.splitext(path_rr2)
    while not os.path.isfile(path_ecg) or file_extension != ".rr2":
        print("Некорректный путь до разметки. Введите ещё раз:")
        path_rr2 = input()
        filename, file_extension = os.path.splitext(path_rr2)
    print("Введите путь до выходного файла формата .csv:")
    path_output = input()
    filename, file_extension = os.path.splitext(path_output)
    arr = path_output.split("/")
    # /large/home/polindromka/project/Automatic-Interpretation-Of-Holter-ECG/final/Holter/result.csv
    path="/".join(arr[:-1])
    while not os.path.isdir(path) or file_extension != ".csv":
        print("Некорректный путь до файла. Введите ещё раз:")
        path_output = input()
        filename, file_extension = os.path.splitext(path_output)
        arr = path_output.split("/")
        path = "/".join(arr[:-1])
    start = time()
    expirement_run(path_ecg, path_rr2, path_output)
    print(f'Done in {int(time() - start)} sec.')


if __name__ == "__main__":
    main()
