import os
import numpy as np
from sklearn.model_selection import KFold
import shutil

# 设置文件夹路径
npz_dir = '/data/xyp/下载/data/npz/FL'  # 替换为你的npz文件夹路径
output_dir = '/data/xyp/下载/data/npz/FL_fold'  # 替换为输出结果文件夹路径

# 获取npz文件列表
npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]

# 创建KFold对象进行五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 创建训练和验证文件夹
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 开始五折交叉验证
fold_idx = 0
for train_idx, val_idx in kf.split(npz_files):
    # 创建当前折的训练集和验证集文件夹
    fold_train_dir = os.path.join(train_dir, f'fold_{fold_idx}')
    fold_val_dir = os.path.join(val_dir, f'fold_{fold_idx}')
    
    os.makedirs(fold_train_dir, exist_ok=True)
    os.makedirs(fold_val_dir, exist_ok=True)
    
    # 获取当前折的训练集和验证集文件
    train_files = [npz_files[i] for i in train_idx]
    val_files = [npz_files[i] for i in val_idx]
    
    # 复制训练集文件到训练集文件夹
    for file in train_files:
        shutil.copy(os.path.join(npz_dir, file), fold_train_dir)
    
    # 复制验证集文件到验证集文件夹
    for file in val_files:
        shutil.copy(os.path.join(npz_dir, file), fold_val_dir)

    print(f"Fold {fold_idx}: {len(train_files)} train files, {len(val_files)} val files")
    fold_idx += 1

print("两折交叉验证数据划分完成！")
