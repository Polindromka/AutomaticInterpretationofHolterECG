import functools
from time import time
import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import homogeneity_score

import data_utils, processing


def valid_split(df, valid_size, seed=None):
    patients = sorted(df['patient'].unique())
    train, valid = train_test_split(patients, test_size=valid_size, random_state=seed)
    train = df[df['patient'].isin(train)]
    valid = df[df['patient'].isin(valid)]
    return train, valid

@functools.lru_cache(maxsize=1)
def get_data(anns_path, ecgs_path):
    anns = pd.read_csv(anns_path)  
    ecgs = list()
    for x in range(5425):
        ecgs.append(np.load(f"{ecgs_path}/data{x}.npy"))
    ecgs=np.concatenate(ecgs)  
    return anns, ecgs


def experiment_v2(params):
    print('Loading data ... ', end='')
    start = time()

    anns = np.load(params['anns_file'])
    anns = pd.DataFrame(anns)
    anns.columns = ['pos_R1', 'tip_qrs', 'patient']
    anns['index'] = range(anns.shape[0])

    ecgs = np.load(params['ecgs_file'])
    print(f'Done in {int(time() - start)} sec.')
    print(f'Anns shape: {anns.shape}')
    print(f'ECGs shape: {ecgs.shape}')
    print()

    print('Splitting train/validation ... ', end='')
    start = time()

    train = anns
    train_ecgs = ecgs[train['index'].values]

    train_ecgs = train_ecgs.reshape(train_ecgs.shape[0], -1)
    print(f'Done in {int(time() - start)} sec.')
    print('Train ECGs shape:', train_ecgs.shape)
    print()

    print('Training embedder ... ', end='')
    start_time = time()
    embedder = params['embedder_class'](**params['embedder_params'])
    embedder.fit(train_ecgs)

    window = params['window']
    halfwindow = window // 2
    result = list()
    test_patients = data_utils.get_test()
    print('Test patients:', len(test_patients))
    for patient in test_patients:

        print('Patient', patient)
        patient_start = time()
        print('Loading data ... ', end='')
        start_time = time()
        ann = data_utils.get_ann(patient, path=data_utils.VOTED_PATH)
        ann = ann[ann['tip_qrs'].isin([0, 2])]
        ann = ann[ann['shum'] == 0]
        if (ann['tip_qrs'] == 2).mean() > 0.1:
            ecg = data_utils.get_ecg(patient)
            end_time = time()
            print(f'Done in {int(end_time - start_time)} seconds.')

            print('Processing ecg ... ', end='')
            start_time = time()
            ecg = processing.process_ecg(ecg).astype('float32')
            ecgs = list()
            goods = list()
            for index, row in ann.iterrows():
                ecg_slice = processing.get_slice(ecg, row['pos_R1'], window, maxval=20, threshold=10)
                if ecg_slice is not None:
                    ecgs.append(ecg_slice)
                    goods.append(True)
                else:
                    goods.append(False)

            if np.any(goods):
                ecgs = np.stack(ecgs)

                ann['goods'] = goods
                end_time = time()
                print(f'Done in {int(end_time - start_time)} seconds.')

                print('Training embedder ... ', end='')
                start_time = time()
                embedder.fit(train_ecgs)
                end_time = time()
                print(f'Done in {int(end_time - start_time)} seconds.')

                print('Embeddings ... ', end='')
                start_time = time()
                embeddings = embedder.transform(ecgs)
                embeddingd_mean = embeddings.mean(axis=0)[None, :]
                embeddings_std = embeddings.std(axis=0)[None, :]
                embeddings -= embeddingd_mean
                embeddings /= embeddings_std
                end_time = time()
                print(f'Done in {int(end_time - start_time)} seconds.')

                print('Clustering ... ', end='')
                start_time = time()
                clusterer = params['clusterer_class'](**params['clusterer_params'])
                clusterer.fit(embeddings)
                labels = clusterer.labels_
                ann['label'] = -1
                ann.loc[ann['goods'], 'label'] = labels
                end_time = time()
                print(f'Done in {int(end_time - start_time)} seconds.')

                n_clusters = np.unique(labels).shape[0] - 1
                homogeneity = homogeneity_score(ann['tip_qrs'], ann['label'])
                noise = (labels == -1).sum() / labels.shape[0]

                row = dict()
                row['patient'] = patient
                row['n_qrs'] = ann.shape[0]
                row['tip_qrs_2_fraq'] = (ann['tip_qrs'] == 2).mean()
                row['n_clusters'] = n_clusters
                row['homogeneity'] = homogeneity
                row['noise'] = noise
                result.append(row)
                pd.DataFrame(result).to_csv(f'logs/pipe_logs_{params["name"]}.csv')
                pd.DataFrame(ann).to_csv(f'{params["output_path"]}')
                patient_end = time()
                print(f'Patietn done in {(patient_end - patient_start)} seconds')
            else:
                print('Done. Bad ecg')

        else:
            print('Done. Low Wide QRS count')

    return n_clusters, homogeneity, noise

def run(params):
    embedder = params['embedder_class'](**params['embedder_params'])
    embedder.net = torch.load(embedder.weights_file)
    ann = data_utils.get_ann(params['patient'], path=params["RR2_path"])
    ecg = data_utils.get_ecg(params['patient'], path=params["ECG_path"])
    ecg = processing.process_ecg(ecg).astype('float32')
    ecgs = list()
    goods = list()
    for index, row in ann.iterrows():
        ecg_slice = processing.get_slice(ecg, row['pos_R1'], params['window'], maxval=20, threshold=10)
        if ecg_slice is not None:
            ecgs.append(ecg_slice)
            goods.append(True)
        else:
            goods.append(False)

    if np.any(goods):
        ecgs = np.stack(ecgs)
        ann['goods'] = goods
        embeddings = embedder.transform(ecgs)
        clusterer = params['clusterer_class'](**params['clusterer_params'])
        clusterer.fit(embeddings)
        labels = clusterer.labels_
        ann['label'] = -1
        for x in range(embeddings.shape[1]):
            ann[f'emb_{x}']=0
            ann.loc[ann['goods'], f'emb_{x}'] = embeddings[:, x]
        ann.loc[ann['goods'], 'label'] = labels
        ann.to_csv(f"{params['output_path']}")
    else:
        print('Done. Bad ecg')