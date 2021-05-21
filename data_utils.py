import numpy as np
import pandas as pd


from glob import glob
import os

ECG_PATH = '/large/datasets/holter/ecg'
RR2_PATH = '/large/datasets/holter/rr2'
VOTED_PATH = '/large/datasets/holter/voted'

ECG_FILE = '/large/datasets/holter/train_ecg_v2.npy'
ANN_FILE = '/large/datasets/holter/train_ann_v2.npy'

def get_patients():
    ecgs = sorted(list(glob(f'{ECG_PATH}/*.ecg')))
    patients = list()
    for ecg_path in ecgs:
        patient = ecg_path.split('/')[-1].split('.')[0]
        if os.path.isfile(f'{RR2_PATH}/{patient}.rr2'):
            patients.append(int(patient))
    return patients

def get_train():
    ecgs = sorted(list(glob(f'{ECG_PATH}/*.ecg')))
    patients = list()
    test = get_test()
    for ecg_path in ecgs:
        patient = int(ecg_path.split('/')[-1].split('.')[0])
        if os.path.isfile(f'{RR2_PATH}/{patient}.rr2'):
            if patient not in test:
                patients.append(patient)
    return patients


def get_test():
    ecgs = sorted(list(glob(f'{ECG_PATH}/*.ecg')))
    patients = list()
    for ecg_path in ecgs:
        patient = int(ecg_path.split('/')[-1].split('.')[0])
        if os.path.isfile(f'{VOTED_PATH}/{patient}.rr2'):
            patients.append(patient)
    return patients

def get_ecg(patient=None, path=None):
    if patient is None and path is None:
        raise FileExistsError
    path = path if path else ECG_PATH
    dt = np.dtype('uint16')
    dt = dt.newbyteorder('>')
    if patient is not None:
        path=f'{path}/{patient}.ecg'
    with open(path, 'rb') as f:
        signal = np.frombuffer(f.read(), dtype=dt)
    signal = signal[:(signal.shape[0] // 3) * 3]
    signal = signal.reshape(-1, 3).T
    signal = signal[:2, :].astype('uint16')
        
    return signal

def get_ann(patient=None, path=None):
    if patient is None and path is None:
        raise FileExistsError
    path = path if path else RR2_PATH
    if patient is not None:
        path = f'{path}/{patient}.ecg'
    with open(path, 'rb') as f:
        signal = np.frombuffer(f.read(), dtype=np.int32)       
    result = decode_ann(signal)       
    return result


def decode_ann(signal, fix=True):
    signal = signal.reshape((-1, 30)).copy()
    result = pd.DataFrame(signal)
    cols = ['nom', 'pos_p1', 'pos_q1', 'pos_R1', 'pos_s1', 'pos_t11', 'pos_t12', 'ampl1', 'ampl1_znak',
           'd_extrem1', 'ST1', 'ST1_znak', 'pos_p2', 'pos_q2', 'pos_R2', 'pos_s2', 'pos_t21', 'pos_t22',
           'ampl2', 'ampl2_znak', 'd_extrem2', 'ST2', 'ST2_znak', 'tip_qrs', 'nom_frm', 'p_art', 'shum',
           'extr_pauz', 'epizod', 'fp']
    result.columns = cols
    
    if fix:
        tip_qrs_map = {-2: 3, -1: 4, 0: 5, 1: 0, 2: 1, 3: 2, 4: 3}
        result['tip_qrs'] = result['tip_qrs'].map(tip_qrs_map)

        extr_pauz_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 10: 0}
        result['extr_pauz'] = result['extr_pauz'].map(extr_pauz_map)
        
    result['tip_qrs'] = result['tip_qrs'].astype('int32')
    result['extr_pauz'] = result['extr_pauz'].astype('int32')
    result['is_pauz'] = result['extr_pauz'].isin([2, 3, 4]).astype('int32')
    
    return result