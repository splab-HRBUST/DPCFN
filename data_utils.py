"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import torch
import collections
import os
import soundfile as sf
import librosa
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed
import h5py
LOGICAL_DATA_ROOT = '/g813_u1/mnt/2021' 
PHISYCAL_DATA_ROOT = '/mnt/2021/'
 
ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDataset(Dataset):
    """ Utility class to load  train/dev datatsets """
    def __init__(self, transform=None, 
        is_train=True, sample_size=None, 
        is_logical=True, feature_name=None, is_eval=False,
        eval_part=0):
        '''
        is_train=False, is_logical=True,
        transform=transforms,
        feature_name=cqcc, 
        is_eval =  False 	 
        eval_part =  0
        '''
        if is_logical:
            data_root = LOGICAL_DATA_ROOT
            track = 'LA'
            #data_root = os.path.join(data_root,track)
        else:
            data_root = PHISYCAL_DATA_ROOT
            track = 'PA'
            data_root = os.path.join(data_root,track)
        if is_eval:
            data_root = os.path.join(data_root,"")
        assert feature_name is not None, 'must provide feature name'
        self.track = track
        self.is_logical = is_logical

        #self.prefix = 'ASVspoof2019_{}'.format(track)
        self.prefix = 'ASVspoof_{}'.format(track)

        v1_suffix = ''
        if is_eval and track == 'PA':
            v1_suffix=''
        self.sysid_dict = {
            '-': 0,  # bonafide speech
            'A01': 1, # Wavenet vocoder
            'A02': 2, # Conventional vocoder WORLD
            'A03': 3, # Conventional vocoder MERLIN
            'A04': 4, # Unit selection system MaryTTS
            'A05': 5, # Voice conversion using neural networks
            'A06': 6, # transform function-based voice conversion
            # For PA:
            'A07':7,
            'A08':8,
            'A09':9,
            'A10':10,
            'A11':11,
            'A12':12,
            'A13':13,
            'A14':14,
            'A15':15,
            'A16':16,
            'A17':17,
            'A18':18,
            'A19':19,
            'AA':20,
            'AB':21,
            'AC':22,
            'BA':23,
            'BB':24,
            'BC':25,
            'CA':26,
            'CB':27,
            'CC':28,


        }
        
        self.is_eval = is_eval
        self.sysid_dict_inv = {v:k for k,v in self.sysid_dict.items()}# 新建一个字典，v和k的值互换。
        self.data_root = data_root
        print("self.data_root = ",self.data_root)
        self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
        self.protocols_fname = 'eval.trl2' if is_eval else 'train.trn' if is_train else 'dev.trl'
        self.protocols_dir = os.path.join(self.data_root,
            '{}_cm_protocols/'.format(self.prefix))
        
        #self.files_dir = os.path.join(self.data_root, '{}_{}'.format(
        #    self.prefix, self.dset_name )+v1_suffix, 'flac')
        self.files_dir = os.path.join(self.data_root, track, 'ASVspoof2019_{}_{}'.format(
            track, self.dset_name )+v1_suffix, 'flac')


        print("self.protocols_dir = ",self.protocols_dir)
        self.protocols_fname = os.path.join(self.protocols_dir,
            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))
        print("self.protocols_fname = ",self.protocols_fname)
        # self.cache_fname = 'cache_{}{}_{}_{}.npy'.format(self.dset_name,
        # '' if is_eval else '',track, feature_name)
        self.cache_fname = '/g813_u1/mkj/twice_attention_networks-main/ta-network-main/cache_{}{}_{}_{}.npy'.format(self.dset_name,
        '' if is_eval else '',track, feature_name)
        self.cache_matlab_fname = 'cache_{}{}_{}_{}.mat'.format(
            self.dset_name, '' if is_eval else '',
             track, feature_name)
        print("self.cache_fname = ",self.cache_fname)
        print("self.cache_matlab_fname = ",self.cache_matlab_fname)
        self.transform = transform
        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname)
            '''
            查看data_x和data_y,与data_sysid与matlab中是否一致。
            '''
            print('Dataset loaded from cache ', self.cache_fname)
        elif feature_name == 'cqcc':
            if os.path.exists(self.cache_matlab_fname):
                self.data_x, self.data_y, self.data_sysid = self.read_matlab_cache(self.cache_matlab_fname)
                '''
                cache_train_LA_cqcc.mat和cache_dev_LA_cqcc.mat和cache_eval_LA_cqcc.mat中:
                data_x  是训练集特征，
                data_y  是语音是否是欺诈语音，
                filename是音频文件名，
                sys_id  是欺诈系统id    
                '''
                self.files_meta = self.parse_protocols_file(self.protocols_fname)
                print('Dataset loaded from matlab cache ', self.cache_matlab_fname)
                torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta),
                           self.cache_fname, pickle_protocol=4)
                print('Dataset saved to cache ', self.cache_fname)
            else:
                print("Matlab cache for cqcc feature do not exist.")
        else:

            self.files_meta = self.parse_protocols_file(self.protocols_fname)
            """
            ASVFile(speaker_id='LA_0039',
            file_name='LA_E_2834763',
            path='/home/g813_u6/2019/LA/ASVspoof2019_LA_eval/flac/LA_E_2834763.flac',
            sys_id=11,
            key=0)的列表
            """
            print("开始处理语音文件~")
            data = list(map(self.read_file, self.files_meta))
            print("语音文件处理完成~")
            """data由特征，标签，欺骗设备id组成"""
            self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
            if self.transform:
                print("开始处理数据集！")
                self.data_x = Parallel(n_jobs=64,prefer='processes')(delayed(self.transform)(x) for x in self.data_x)
                print("数据集处理完成！")
            torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
            print('Dataset saved to cache ', self.cache_fname)
        if sample_size: # sample_size =  None
            select_idx = np.random.choice(len(self.files_meta), size=(sample_size,), replace=True).astype(np.int32)
            self.files_meta= [self.files_meta[x] for x in select_idx]
            self.data_x = [self.data_x[x] for x in select_idx]
            self.data_y = [self.data_y[x] for x in select_idx]
            self.data_sysid = [self.data_sysid[x] for x in select_idx]
        self.length = len(self.data_x)


        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.files_meta[idx]

    def read_file(self, meta):
        data_x, sample_rate = sf.read(meta.path)
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        
        
        if self.is_eval:
            #return ASVFile(speaker_id='',
            #file_name=tokens[0],
            #path=os.path.join(self.files_dir, tokens[0] + '.flac'),
            #sys_id=0,
            #key=0)

            return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))
        
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))
        
    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)  
        return list(files_meta)

    def read_matlab_cache(self, filepath):
        f = h5py.File(filepath, 'r')
        # filename_index = f["filename"]
        # filename = []
        data_x_index = f["data_x"]
        sys_id_index = f["sys_id"]
        data_x = []
        data_y = f["data_y"][0]
        sys_id = []
        for i in range(0, data_x_index.shape[1]):
            idx = data_x_index[0][i]  # data_x
            temp = f[idx]
            data_x.append(np.array(temp).transpose())
            # idx = filename_index[0][i]  # filename
            # temp = list(f[idx])
            # temp_name = [chr(x[0]) for x in temp]
            # filename.append(''.join(temp_name))
            idx = sys_id_index[0][i]  # sys_id
            temp = f[idx]
            sys_id.append(int(list(temp)[0][0]))
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        return data_x.astype(np.float32), data_y.astype(np.int64), sys_id


# if __name__ == '__main__':
#    train_loader = ASVDataset(LOGICAL_DATA_ROOT, is_train=True)
#    assert len(train_loader) == 25380, 'Incorrect size of training set.'
#    dev_loader = ASVDataset(LOGICAL_DATA_ROOT, is_train=False)
#    assert len(dev_loader) == 24844, 'Incorrect size of dev set.'

