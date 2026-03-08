# %% move files based on csv file
import numpy as np
import nibabel as nb
import os
from os import listdir, makedirs
from os.path import basename, join, dirname, isfile, isdir
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import multiprocessing as mp
import argparse

def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)



parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seg_dir', default='test_demo/segs', type=str)
parser.add_argument('-g', '--gt_dir', default='test_demo/gts', type=str)
parser.add_argument('-csv_dir', default='test_demo/metrics.csv', type=str)
parser.add_argument('-num_workers', type=int, default=1)
args = parser.parse_args()

seg_dir = args.seg_dir
gt_dir = args.gt_dir
csv_dir = args.csv_dir
num_workers = args.num_workers

def compute_metrics(npz_name):
    metric_dict = {'dsc': -1.}
    
    npz_seg = np.load(join(seg_dir, npz_name), allow_pickle=True, mmap_mode='r')
    npz_gt = np.load(join(gt_dir, npz_name), allow_pickle=True, mmap_mode='r')
    gts = npz_gt['gts']
    segs = npz_seg['segs']
    if npz_name.startswith('3D'):
        spacing = npz_gt['spacing']
    
    # print(npz_name)
    # print(gts.shape)
    # print(segs.shape)
    if gts.shape!=segs.shape:
        dsc=-1
    else:
        dsc = compute_multi_class_dsc(gts, segs)
    #print(dsc)

    #print(np.squeeze(npz_name, dsc, nsd))
    return npz_name, dsc

if __name__ == '__main__':
    seg_metrics = OrderedDict()
    seg_metrics['case'] = []
    seg_metrics['dsc'] = []
    
    npz_names = listdir(gt_dir)
    npz_names = [npz_name for npz_name in npz_names if npz_name.endswith('.npz')]
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(npz_names)) as pbar:
            for i, (npz_name, dsc) in enumerate(pool.imap_unordered(compute_metrics, npz_names)):
                seg_metrics['case'].append(npz_name)
                seg_metrics['dsc'].append(np.round(dsc, 4))
                pbar.update()
    # 计算平均dice
    average_dsc = np.mean(seg_metrics['dsc'])
    
    # 将平均值添加为新的行
    seg_metrics['case'].append('Average')
    seg_metrics['dsc'].append(np.round(average_dsc, 4))
    
    # 创建DataFrame并按'case'排序
    df = pd.DataFrame(seg_metrics)
    df = df.sort_values(by=['case'])
    
    # 保存结果到CSV文件
    df.to_csv(csv_dir, index=False)