# %%
import os
import multiprocessing
import random
import monai
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
import cv2
import torch.nn.functional as F

from matplotlib import pyplot as plt
import argparse

from evaluation.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
# from evaluation.compute_metrics import compute_multi_class_nsd

from lightweight_detector import load_detector_model, generate_lesion_bbox


def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
        #print(np.mean(dsc))
    return np.mean(dsc)

def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i
        gt_i=np.squeeze(gt_i)
        seg_i=np.squeeze(seg_i)
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.mean(nsd)


# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "-data_root", type=str, default="/data/xyp/下载/data/npy/FL/train_0",
    help="Path to the npy data root."
)
parser.add_argument(
    # "-pretrained_checkpoint", type=str, default="lite_medsam.pth",
    "-pretrained_checkpoint", type=str, default=None,
    help="Path to the pretrained Lite-MedSAM checkpoint."
)
parser.add_argument(
    "-resume", type=str, default='workdir_FL_full/medsam_lite_best.pth',
    help="Path to the checkpoint to continue training."
)
parser.add_argument(
    "-work_dir", type=str, default="./workdir_FL_full",
    help="Path to the working directory where checkpoints and logs will be saved."
)
parser.add_argument(
    "-num_epochs", type=int, default=100,
    help="Number of epochs to train."
)
parser.add_argument(
    "-batch_size", type=int, default=4,
    help="Batch size."
)
parser.add_argument(
    "-num_workers", type=int, default=8,
    help="Number of workers for dataloader."
)
parser.add_argument(
    "-device", type=str, default="cuda:2",
    help="Device to train on."
)
parser.add_argument(
    "-bbox_shift", type=int, default=5,
    help="Perturbation to bounding box coordinates during training."
)
parser.add_argument(
    "-lr", type=float, default=0.00008,
    help="Learning rate."
)
parser.add_argument(
    "-weight_decay", type=float, default=0.01,
    help="Weight decay."
)
parser.add_argument(
    "-iou_loss_weight", type=float, default=1.0,
    help="Weight of IoU loss."
)
parser.add_argument(
    "-seg_loss_weight", type=float, default=1.0,
    help="Weight of segmentation loss."
)
parser.add_argument(
    "-ce_loss_weight", type=float, default=1.0,
    help="Weight of cross entropy loss."
)
parser.add_argument(
    "--sanity_check", action="store_true",
    help="Whether to do sanity check for dataloading."
)
parser.add_argument(
    "-test_pathfile", type=str, default="/data/xyp/下载/data/npz/FL/val_0",
    help="file path of test every 10 epochs."
)
parser.add_argument(
    "-train_pathfile", type=str, default="/data/xyp/下载/data/npz/FL/train_0",
    help="file path of train every 10 epochs."
)
parser.add_argument(
    "-test_data_root", type=str, default="/data/xyp/下载/data/npy/FL/val_0",
    help="file path of train every 10 epochs."
)

args = parser.parse_args()

if __name__ == "__main__":

    # %%
    work_dir = args.work_dir
    data_root = args.data_root
    medsam_lite_checkpoint = args.pretrained_checkpoint
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = args.device
    bbox_shift = args.bbox_shift
    lr = args.lr
    weight_decay = args.weight_decay
    iou_loss_weight = args.iou_loss_weight
    seg_loss_weight = args.seg_loss_weight
    ce_loss_weight = args.ce_loss_weight
    do_sancheck = args.sanity_check
    checkpoint = args.resume
    test_pathfile=args.test_pathfile
    train_pathfile=args.train_pathfile
    best_val_loss = 1e10
    patience=10
    test_data_root=args.test_data_root

    makedirs(work_dir, exist_ok=True)

    import wandb
    os.environ["WANDB_MODE"] = "online"
    #swanlab.init(project="MedSAM_sarcoma_box", name="lymphoma_train_medsam_box0",config={
    #swanlabab.init(project="MedSAM_sarcoma_box", name="lymphoma_train_fromscratch",config={
    #swanlab.init(project="MedSAM_sarcoma_box", name="2fold_sarcoma_pretrain_fromscratch1",config={
    wandb.init(project="DLBCL2FL", name="FL_full",config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": "Adam"
    })
    # %%
    torch.cuda.empty_cache()
    os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6


    def freeze_encoder_layers(encoder, 
                            freeze_patch_embed=False,
                            freeze_layers_idx=None,
                            freeze_neck=False):
        """
        冻结图像编码器中的指定组件
        Args:
            encoder: TinyViT 图像编码器实例
            freeze_patch_embed: 是否冻结初始的patch_embed层
            freeze_layers_idx: 要冻结的层索引列表 (e.g., [0,1] 冻结前两层)
            freeze_neck: 是否冻结最后的neck层
        """
        # 1. 冻结初始patch_embed层
        if freeze_patch_embed:
            print(f"Freezing patch_embed")
            for param in encoder.patch_embed.parameters():
                param.requires_grad = False
        
        # 2. 冻结指定层
        if freeze_layers_idx is not None:
            for i in freeze_layers_idx:
                if i < len(encoder.layers):
                    print(f"Freezing layer {i}")
                    for param in encoder.layers[i].parameters():
                        param.requires_grad = False
        
        # 3. 冻结最后的neck层
        if freeze_neck:
            print(f"Freezing neck")
            for param in encoder.neck.parameters():
                param.requires_grad = False


    def cal_iou(result, reference):
        
        intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
        union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
        
        iou = intersection.float() / union.float()
        
        return iou.unsqueeze(1)

    # %%
    class NpyDataset(Dataset): 
        def __init__(self, data_root, image_size=256, bbox_shift=5, data_aug=False):
            self.data_root = data_root
            self.gt_path = join(data_root, "gts")
            self.img_path = join(data_root, "imgs")
            self.gt_path_files = sorted(glob(join(self.gt_path, '*.npy'), recursive=True))
            self.gt_path_files = [
                file for file in self.gt_path_files
                if isfile(join(self.img_path, basename(file)))
            ]
            self.image_size = image_size
            self.target_length = image_size
            self.bbox_shift = bbox_shift
            self.data_aug = data_aug
        
        def __len__(self):
            return len(self.gt_path_files)

        def __getitem__(self, index):
            img_name = basename(self.gt_path_files[index])
            assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
            img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
            img_resize = self.resize_longest_side(img_3c)
            # Resizing
            img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
            img_padded = self.pad_image(img_resize) # (256, 256, 3)
            # convert the shape to (3, H, W)
            img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
            assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
            gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
            gt = cv2.resize(
                gt,
                (img_resize.shape[1], img_resize.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            gt = self.pad_image(gt) # (256, 256)
            label_ids = np.unique(gt)[1:]
            try:
                gt2D = np.uint8(gt == random.choice(label_ids.tolist())) # only one label, (256, 256)
            except:
                print(img_name, 'label_ids.tolist()', label_ids.tolist())
                gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)
            
            #print(img_padded.shape)
            # add data augmentation: random fliplr and random flipud
            if self.data_aug:
                gt2D_backup = gt2D.copy()
                if random.random() > 0.5:
                    img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                    gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                    # print('DA with flip left right')
                if random.random() > 0.5:
                    img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                    gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                    # print('DA with flip upside down')
                if not gt2D.any():
                    gt2D=gt2D_backup.copy()
                else:
                    gt2D_backup=gt2D.copy()

            if self.data_aug:
                # --- 几何变换增强 ---
                # 2. 随机旋转（-10°~10°）
                if random.random() > 0.5:
                    angle = random.uniform(-10, 10)
                    img_padded = rotate(img_padded, angle, axes=(1,0), reshape=False, mode='reflect')
                    gt2D = rotate(gt2D, angle, axes=(1,0), reshape=False, mode='reflect', order=0)  # order=0 保持标注二值性
                if not gt2D.any():
                    gt2D=gt2D_backup.copy()
                else:
                    gt2D_backup=gt2D.copy()

                # 3. 随机弹性形变（模拟解剖结构形变）
                if random.random() > 0.3:
                    c, h, w = img_padded.shape  # (3, 256, 256)
                    alpha = random.randint(5, 20)
                    sigma = random.randint(5, 10)
                    
                    # 生成 2D 位移场（基于空间维度 H, W）
                    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
                    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
                    
                    # 生成网格坐标
                    x, y = np.meshgrid(np.arange(w), np.arange(h))
                    indices = (y + dy).reshape(-1, 1), (x + dx).reshape(-1, 1)
                    
                    # 标注同步形变（单通道）
                    gt1 = map_coordinates(gt2D, indices, order=0, mode='reflect').reshape(h, w)
                    gt1 = np.uint8(gt1 > 0)  # 二值化
                    if gt1.any():
                        gt2D=gt1
                        # 对每个通道应用相同的形变
                        for ch in range(c):
                            img_padded[ch] = map_coordinates(
                                img_padded[ch], indices, order=3, mode='reflect'
                            ).reshape(h, w)
                if not gt2D.any():
                    gt2D=gt2D_backup.copy()
                else:
                    gt2D_backup=gt2D.copy()
                    


                # --- 强度扰动增强 ---
                # 4. 添加高斯噪声（模拟PET低剂量噪声）
                if random.random() > 0.5:
                    noise_sigma = random.uniform(0.01, 0.05) * np.max(img_padded)
                    noise = np.random.normal(0, noise_sigma, img_padded.shape)
                    # img_padded = np.clip(img_padded + noise, 0, 255)  # 保持像素范围
                    img_padded = np.clip(img_padded + noise, 0.0, 1.0)
                if not gt2D.any():
                    gt2D=gt2D_backup.copy()
                else:
                    gt2D_backup=gt2D.copy()

                # 5. 随机亮度/对比度调整
                if random.random() > 0.5:
                    alpha = random.uniform(0.9, 1.1)  # 对比度系数
                    beta = random.uniform(-0.1, 0.1)    # 亮度偏移
                    # img_padded = np.clip(alpha * img_padded + beta, 0, 255)
                    img_padded = np.clip(alpha * img_padded + beta, 0.0, 1.0)
                if not gt2D.any():
                    gt2D=gt2D_backup.copy()
                else:
                    gt2D_backup=gt2D.copy()

                # 6. 模拟部分容积效应（模糊）
                if random.random() > 0.5:
                    sigma = random.uniform(0.3,0.8)
                    img_padded = gaussian_filter(img_padded, sigma=sigma)
                    img_padded = np.clip(img_padded, 0.0, 1.0)
                if not gt2D.any():
                    gt2D=gt2D_backup.copy()
                else:
                    gt2D_backup=gt2D.copy()

                # --- 高级增强（医学影像特异性）---
                # 7. 随机遮挡（模拟病灶被部分遮挡）
                # if random.random() > 0.3:
                #     c, h, w = img_padded.shape  # 输入形状为 (3, 256, 256)
                #     num_occlusions = random.randint(1, 3)
                    
                #     for _ in range(num_occlusions):
                #         # 随机生成遮挡区域大小（基于空间维度 h,w）
                #         occl_h = random.randint(int(0.05*h), int(0.1*h))
                #         occl_w = random.randint(int(0.05*w), int(0.1*w))
                        
                #         # 随机生成遮挡位置（空间坐标）
                #         x = random.randint(0, w - occl_w)
                #         y = random.randint(0, h - occl_h)
                        
                #         # 在所有通道的相同位置添加遮挡
                #         noise_value = np.random.normal(0, 0.2)  # 基于 [0,1] 归一化范围
                #         img_padded[:, y:y+occl_h, x:x+occl_w] = np.clip(
                #             np.random.normal(0, 0.2, (c, occl_h, occl_w)),  # 生成 3D 噪声
                #             0, 1  # 保持 [0,1] 范围
                #         )

                # 8. 随机仿射变换（缩放+平移）
                # if random.random() > 0.5:
                #     c, h, w = img_padded.shape  # 关键修复：显式定义 h, w
                #     scale = random.uniform(0.9, 1.1)
                #     tx = random.randint(-int(0.05*w), int(0.05*w))
                #     ty = random.randint(-int(0.05*h), int(0.05*h))
                #     M = np.array([[scale, 0, tx],
                #                 [0, scale, ty]], dtype=np.float32)
                    
                #     # 修复步骤：处理 CHW -> HWC 转换
                #     # Step 1: 转换图像为 HWC 格式
                #     img_hwc = np.transpose(img_padded, (1, 2, 0))  # (256, 256, 3)
                    
                #     # Step 2: 应用仿射变换
                #     img_hwc = cv2.warpAffine(
                #         img_hwc, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                #     )
                    
                #     # Step 3: 转换回 CHW 格式
                #     img_padded = np.transpose(img_hwc, (2, 0, 1))  # (3, 256, 256)
                    
                #     # 处理标注（保持 2D）
                #     gt2D = cv2.warpAffine(
                #         gt2D, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT
                #     )
            gt2D = np.uint8(gt2D > 0)


            # y_indices, x_indices = np.where(gt2D > 0)
            # x_min, x_max = np.min(x_indices), np.max(x_indices)
            # y_min, y_max = np.min(y_indices), np.max(y_indices)
            # # add perturbation to bounding box coordinates
            # H, W = gt2D.shape
            # x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            # x_max = min(W-1, x_max + random.randint(0, self.bbox_shift))
            # y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            # y_max = min(H-1, y_max + random.randint(0, self.bbox_shift))
            # bboxes = np.array([x_min, y_min, x_max, y_max])

            # 全局勾画：使用整个图像范围作为边界框（无扰动，覆盖全图）
            # H, W = gt2D.shape  # gt2D已被处理为256x256（与图像尺寸一致）
            # x_min = 0  # 左上角x坐标
            # y_min = 0  # 左上角y坐标
            # x_max = W - 1  # 右下角x坐标（图像宽度-1）
            # y_max = H - 1  # 右下角y坐标（图像高度-1）
            # bboxes = np.array([x_min, y_min, x_max, y_max])  # 全局边界框

            bboxes = get_bbox(img_padded, gt2D)

            return {
                "image": torch.tensor(img_padded).float(),
                "gt2D": torch.tensor(gt2D[None, :,:]).float(),
                "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
                "image_name": img_name,
                "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
                "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
            }

        def resize_longest_side(self, image):
            """
            Expects a numpy array with shape HxWxC in uint8 format.
            """
            long_side_length = self.target_length
            oldh, oldw = image.shape[0], image.shape[1]
            scale = long_side_length * 1.0 / max(oldh, oldw)
            newh, neww = oldh * scale, oldw * scale
            neww, newh = int(neww + 0.5), int(newh + 0.5)
            target_size = (neww, newh)

            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        def pad_image(self, image):
            """
            Expects a numpy array with shape HxWxC in uint8 format.
            """
            # Pad
            h, w = image.shape[0], image.shape[1]
            padh = self.image_size - h
            padw = self.image_size - w
            if len(image.shape) == 3: ## Pad image
                image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
            else: ## Pad gt mask
                image_padded = np.pad(image, ((0, padh), (0, padw)))

            return image_padded


    import torch


    def resize_longest_side(image, target_length):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(image, target_size):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = target_size - h
        padw = target_size - w
        if len(image.shape) == 3: ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else: ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded

    class MedSAM_Lite(nn.Module):
        def __init__(
                self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder
            ):
            super().__init__()
            self.image_encoder = image_encoder
            self.mask_decoder = mask_decoder
            self.prompt_encoder = prompt_encoder

        def forward(self, image, box_np):
            image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
            # do not compute gradients for prompt encoder
            with torch.no_grad():
                box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :] # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_np,
                masks=None,
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding, # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            ) # (B, 1, 256, 256)

            return low_res_masks,iou_predictions

        @torch.no_grad()
        def postprocess_masks(self, masks, new_size, original_size):
            """
            Do cropping and resizing

            Parameters
            ----------
            masks : torch.Tensor
                masks predicted by the model
            new_size : tuple
                the shape of the image after resizing to the longest side of 256
            original_size : tuple
                the original shape of the image

            Returns
            -------
            torch.Tensor
                the upsampled mask to the original size
            """
            # Crop
            masks = masks[..., :new_size[0], :new_size[1]]
            # Resize
            masks = F.interpolate(
                masks,
                size=(original_size[0], original_size[1]),
                mode="bilinear",
                align_corners=False,
            )

            return masks


    def show_mask(mask, ax, mask_color=None, alpha=0.5):
        if mask_color is not None:
            color = np.concatenate([mask_color, np.array([alpha])], axis=0)
        else:
            color = np.array([251/255, 252/255, 30/255, alpha])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_box(box, ax, edgecolor='blue'):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


    def resize_box(box, new_size, original_size):
        """
        Revert box coordinates from scale at 256 to original scale

        Parameters
        ----------
        box : np.ndarray
            box coordinates at 256 scale
        new_size : tuple
            Image shape with the longest edge resized to 256
        original_size : tuple
            Original image shape

        Returns
        -------
        np.ndarray
            box coordinates at original scale
        """
        new_box = np.zeros_like(box)
        ratio = max(original_size) / max(new_size)
        for i in range(len(box)):
            new_box[i] = int(box[i] * ratio)

        return new_box


    @torch.no_grad()
    def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
        box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
        
        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points = None,
            boxes = box_torch,
            masks = None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False
        )

        low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
        low_res_pred = torch.sigmoid(low_res_pred)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

        return medsam_seg

    def get_bbox(img, gt2D, bbox_shift=5):
        # assert np.max(gt2D)==1 and np.min(gt2D)==0.0, f'ground truth should be 0, 1, but got {np.unique(gt2D)}'
        # y_indices, x_indices = np.where(gt2D > 0)
        # x_min, x_max = np.min(x_indices), np.max(x_indices)
        # y_min, y_max = np.min(y_indices), np.max(y_indices)
        # # add perturbation to bounding box coordinates
        # H, W = gt2D.shape
        # x_min = max(0, x_min - bbox_shift)
        # x_max = min(W, x_max + bbox_shift)
        # y_min = max(0, y_min - bbox_shift)
        # y_max = min(H, y_max + bbox_shift)
        # bboxes = np.array([x_min, y_min, x_max, y_max])
        # return bboxes

        # 全局勾画：直接使用图像完整尺寸作为边界框
        H, W = gt2D.shape  # gt2D已预处理为256x256（与评估图像尺寸一致）
        x_min = 0  # 左上角x坐标
        y_min = 0  # 左上角y坐标
        x_max = W - 1  # 右下角x坐标（255 for 256x256）
        y_max = H - 1  # 右下角y坐标（255 for 256x256）
        return np.array([x_min, y_min, x_max, y_max])




    def MedSAM_infer_npz(gt_path_files):
        all_dices=0
        all_nsds=0
        item=0
        for filename in os.listdir(gt_path_files):
            if filename.endswith('.npz'):
                gt_path_file = os.path.join(gt_path_files, filename)
                #print(gt_path_file)
                    
                npz_data = np.load(gt_path_file, 'r', allow_pickle=True) # (H, W, 3)
                img_3D = npz_data['imgs'] # (Num, H, W)
                gt_3D = npz_data['gts'] # (Num, H, W)
                spacing = npz_data['spacing']
                seg_3D = np.zeros_like(gt_3D, dtype=np.uint8) # (Num, H, W)
                box_list = [dict() for _ in range(img_3D.shape[0])]

                for i in range(img_3D.shape[0]):
                    img_2d = img_3D[i,:,:] # (H, W)
                    H, W = img_2d.shape[:2]
                    img_3c = np.repeat(img_2d[:,:, None], 3, axis=-1) # (H, W, 3)

                    ## MedSAM Lite preprocessing
                    img_256 = resize_longest_side(img_3c, 256)
                    newh, neww = img_256.shape[:2]
                    img_256 = (img_256 - img_256.min()) / np.clip(
                        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
                    )
                    img_256_padded = pad_image(img_256, 256)
                    img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

                    gt = gt_3D[i,:,:] # (H, W)
                    label_ids = np.unique(gt)[1:]
                    for label_id in label_ids:
                        gt2D = np.uint8(gt == label_id) # only one label, (H, W)
                        if gt2D.shape != (newh, neww):
                            gt2D_resize = cv2.resize(
                                gt2D.astype(np.uint8), (neww, newh),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(np.uint8)
                        else:
                            gt2D_resize = gt2D.astype(np.uint8)
                        gt2D_padded = pad_image(gt2D_resize, 256) ## (256, 256)
                        if np.sum(gt2D_padded) > 0:
                            box = get_bbox(img_256_padded, gt2D_padded, bbox_shift) # (4,)
                            sam_mask = medsam_inference(medsam_lite_model, image_embedding, box, (newh, neww), (H, W))
                            #if seg_3D.shape[1]!=144:
                            #    print("dimension wrong!")
                            seg_3D[i, sam_mask>0] = label_id
                            box_list[i][label_id] = box

                label_ids = np.unique(gt_3D)[1:]
                all_dices=all_dices+compute_multi_class_dsc(gt_3D, seg_3D)
                all_nsds=all_nsds+compute_multi_class_nsd(gt_3D, seg_3D,spacing)
                item=item+1
                # np.savez_compressed(
                #     join(pred_save_dir, task_folder, npz_name),
                #     segs=seg_3D, gts=gt_3D, spacing=spacing
                # )
        return all_dices/item, all_nsds/item




    #%% sanity test of dataset class
    if do_sancheck:
        tr_dataset = NpyDataset(data_root, data_aug=False)
        tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
        for step, batch in enumerate(tr_dataloader):
            # show the example
            _, axs = plt.subplots(1, 2, figsize=(10, 10))
            idx = random.randint(0, 4)

            image = batch["image"]
            gt = batch["gt2D"]
            bboxes = batch["bboxes"]
            names_temp = batch["image_name"]

            axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
            show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
            show_box(bboxes[idx].numpy().squeeze(), axs[0])
            axs[0].axis('off')
            # set title
            axs[0].set_title(names_temp[idx])
            idx = random.randint(4, 7)
            axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
            show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
            show_box(bboxes[idx].numpy().squeeze(), axs[1])
            axs[1].axis('off')
            # set title
            axs[1].set_title(names_temp[idx])
            plt.subplots_adjust(wspace=0.01, hspace=0)
            plt.savefig(
                join(work_dir, 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
                bbox_inches='tight',
                dpi=300
            )
            plt.close()
            break

    # %%
    # class MedSAM_Lite(nn.Module):
    #     def __init__(self, 
    #                 image_encoder, 
    #                 mask_decoder,
    #                 prompt_encoder
    #                 ):
    #         super().__init__()
    #         self.image_encoder = image_encoder
    #         self.mask_decoder = mask_decoder
    #         self.prompt_encoder = prompt_encoder
            
    #     def forward(self, image, boxes):
    #         image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

    #         sparse_embeddings, dense_embeddings = self.prompt_encoder(
    #             points=None,
    #             boxes=boxes,
    #             masks=None,
    #         )
    #         low_res_masks, iou_predictions = self.mask_decoder(
    #             image_embeddings=image_embedding, # (B, 256, 64, 64)
    #             image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
    #             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
    #             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
    #             multimask_output=False,
    #           ) # (B, 1, 256, 256)

    #         return low_res_masks, iou_predictions

    #     @torch.no_grad()
    #     def postprocess_masks(self, masks, new_size, original_size):
    #         """
    #         Do cropping and resizing
    #         """
    #         # Crop
    #         masks = masks[:, :, :new_size[0], :new_size[1]]
    #         # Resize
    #         masks = F.interpolate(
    #             masks,
    #             size=(original_size[0], original_size[1]),
    #             mode="bilinear",
    #             align_corners=False,
    #         )

    #         return masks

    # %%

    medsam_lite_image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64, ## (64, 256, 256)
            128, ## (128, 128, 128)
            160, ## (160, 64, 64)
            320 ## (320, 64, 64) 
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.1,
        drop_path_rate=0.2,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=1
        #layer_lr_decay=0.6
    )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )


    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )

    medsam_lite_model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )

    if medsam_lite_checkpoint is not None:
        if isfile(medsam_lite_checkpoint):
            print(f"Finetuning with pretrained weights {medsam_lite_checkpoint}")
            medsam_lite_ckpt = torch.load(
                medsam_lite_checkpoint,
                map_location="cpu"
            )
            if medsam_lite_checkpoint.find("best")==-1:
                medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=True)
            else:
                medsam_lite_model.load_state_dict(medsam_lite_ckpt['model'], strict=True)
        else:
            print(f"Pretrained weights {medsam_lite_checkpoint} not found, training from scratch")

    #freeze prompt_encoder
    # for param in medsam_lite_model.prompt_encoder.parameters():
    #     param.requires_grad = False

    # freeze_encoder_layers(
    #     medsam_lite_model.image_encoder,
    #     freeze_patch_embed=False,
    #     freeze_layers_idx=[],
    #     freeze_neck=False
    # )
    #ensure already encoder parameters
    # encoder_requires_grad = [p.requires_grad for p in medsam_lite_model.image_encoder.parameters()]
    # print(f"编码器参数是否冻结：{not any(encoder_requires_grad)}")  # 应该输出 True
    # decoder_requires_grad = [p.requires_grad for p in medsam_lite_model.mask_decoder.parameters()]
    # print(f"解码器参数是否可训练：{any(decoder_requires_grad)}")    # 应该输出 True

    def init_weights(m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    medsam_lite_model.image_encoder.apply(init_weights)

    medsam_lite_model = medsam_lite_model.to(device)
    medsam_lite_model.train()


    # %%
    optimizer = optim.AdamW(
        medsam_lite_model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
    )
    # optimizer = optim.AdamW(
    #     filter(lambda p: p.requires_grad, medsam_lite_model.parameters()),
    #     lr=lr,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     weight_decay=weight_decay,
    # )


    # optimizer = optim.AdamW(
    #     [
    #         {"params": filter(lambda p: p.requires_grad, medsam_lite_model.image_encoder.parameters()), "weight_decay": 0.02},  # 编码器强正则
    #         {"params": medsam_lite_model.mask_decoder.parameters(), "weight_decay": 0.02},   # 解码器中正则
    #         {"params": medsam_lite_model.prompt_encoder.parameters(), "weight_decay": 0.02},
    #     ],
    #     lr=lr,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    # )

    # # 冻结前总参数量
    total_params = sum(p.numel() for p in medsam_lite_model.parameters())
    print(f"模型总参数量：{total_params:,}")
    total_params = sum(p.numel() for p in medsam_lite_model.mask_decoder.parameters())
    print(f"maskdecoder参数量：{total_params:,}")
    total_params = sum(p.numel() for p in medsam_lite_model.prompt_encoder.parameters())
    print(f"promptencoder参数量：{total_params:,}")
    total_params = sum(p.numel() for p in medsam_lite_model.image_encoder.parameters())
    print(f"imageencoder参数量：{total_params:,}")

    # # 优化器实际管理的可训练参数量
    # trainable_params = sum(p.numel() for p in medsam_lite_model.parameters() if p.requires_grad)
    # print(f"可训练参数量：{trainable_params:,}")  # 应该远小于总参数量

    # # 验证冻结比例是否合理
    # expected_ratio = trainable_params / total_params
    # print(f"可训练参数比例：{expected_ratio:.2%}")


    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.8,
    #     patience=3,
    #     cooldown=0
    # )
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    #seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=False, reduction='mean')
    # weights = torch.tensor([1, 10])
    # ce_loss = nn.BCEWithLogitsLoss(reduction='mean',weight=weights)
    iou_loss = nn.MSELoss(reduction='mean')
    focal_loss=monai.losses.FocalLoss(gamma=5,alpha=0.75)
    # %%
    train_dataset = NpyDataset(data_root=data_root, data_aug=True)
    #train_dataset = NpyDataset(data_root=data_root, data_aug=False)
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers, 
                              pin_memory=True)
    val_dataset = NpyDataset(data_root=test_data_root, data_aug=False)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory=True)

    if checkpoint and isfile(checkpoint):
        print(f"Resuming from checkpoint {checkpoint}")
        checkpoint = torch.load(checkpoint)
        medsam_lite_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = 1e10
    # %%
    train_losses = []
    medsam_lite_model.eval()
    dice_score1,nsd1=MedSAM_infer_npz(train_pathfile)
    print(f"Epoch {0}: Train Dice Score = {dice_score1:.4f}, Train_NSD = {nsd1:.4f}")
    wandb.log({"Train Dice Score": dice_score1,"Train_NSD": nsd1,"Epoch":0})  # 记录到 wandb
    dice_score,nsd2=MedSAM_infer_npz(test_pathfile)
    print(f"Epoch {0}: Validation Dice Score = {dice_score:.4f},Val_NSD = {nsd2:.4f}")
    wandb.log({"Validation Dice Score": dice_score,"Val_NSD": nsd2, "Epoch":0})  # 记录到 wandb
    medsam_lite_model.train()
    iou=[]
    ce=[]
    seg=[]
    epochs_without_improvement=0
    for epoch in range(start_epoch + 1, num_epochs):
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        iou_los = [1e10 for _ in range(len(train_loader))]
        ce_los = [1e10 for _ in range(len(train_loader))]
        seg_los = [1e10 for _ in range(len(train_loader))]
        # focal_los=[1e10 for _ in range(len(train_loader))]
        epoch_start_time = time()
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            image = batch["image"]
            gt2D = batch["gt2D"]
            boxes = batch["bboxes"]
            optimizer.zero_grad()
            image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
            logits_pred, iou_pred = medsam_lite_model(image, boxes)
            l_seg = seg_loss(logits_pred, gt2D)
            
            # l_ce = ce_loss(logits_pred, gt2D.float())
            #mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
            iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            l_iou = iou_loss(iou_pred, iou_gt)
            #loss = mask_loss + iou_loss_weight * l_iou
            #epoch_loss[step] = loss.item()
            l_seg = seg_loss(logits_pred, gt2D)
            l_focal = focal_loss(logits_pred, gt2D)*2
            loss=l_seg #+l_ce#+l_focal
            epoch_loss[step]=loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
            #iou_los[step]=l_iou.item()
            # ce_los[step]=l_ce.item()
            seg_los[step]=l_seg.item()
            # focal_los[step]=l_focal.item()

            
        epoch_end_time = time()
        epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
        train_losses.append(epoch_loss_reduced)
        #lr_scheduler.step(epoch_loss_reduced)
        lr_scheduler.step()
        model_weights = medsam_lite_model.state_dict()
        checkpoint = {
            "model": model_weights,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "loss": epoch_loss_reduced,
            "best_loss": best_loss,
        }
        torch.save(checkpoint, join(work_dir, "medsam_lite_latest.pth"))
        wandb.log({
                    "Train Loss": epoch_loss_reduced,
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                    "Epoch": epoch
                })
        
        wandb.log({
            #"Train iou Loss": sum(iou_los)/len(iou_los),
            # "Train ce loss": sum(ce_los)/len(ce_los),
            "Train seg_loss": sum(seg_los)/len(seg_los),
            # "Train focal_loss": sum(focal_los)/len(focal_los),
            "Epoch":epoch
        })
        if epoch_loss_reduced < best_loss:
            print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
            best_loss = epoch_loss_reduced
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, join(work_dir, "medsam_lite_best.pth"))

        epoch_loss_reduced = 1e10
        # %% plot loss
        plt.plot(train_losses)
        # plt.title("Dice + Binary Cross Entropy + IoU Loss")
        plt.title("Dice + focal Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(work_dir, "train_loss.png"))
        plt.close()
        if (epoch + 1) % 1 == 0:     
            medsam_lite_model.eval()  # 切换到评估模式
            iou_lo = [1e10 for _ in range(len(val_loader))]
            # ce_los = [1e10 for _ in range(len(val_loader))]
            seg_lo = [1e10 for _ in range(len(val_loader))]
            # focal_lo=[1e10 for _ in range(len(val_loader))]
            val_loss = 0.0
            with torch.no_grad():
                if train_pathfile!=None:
                    dice_score1,nsd1=MedSAM_infer_npz(train_pathfile)
                    print(f"Epoch {epoch + 1}: Train Dice Score = {dice_score1:.4f}, Train_NSD= {nsd1:.4f}")
                    wandb.log({"Train Dice Score": dice_score1,"Train_NSD": nsd1, "Epoch":epoch})  # 记录到 wandb
                dice_score2, nsd2=MedSAM_infer_npz(test_pathfile)
                print(f"Epoch {epoch + 1}: Validation Dice Score = {dice_score2:.4f}, Val_NSD= {nsd2:.4f}")
                wandb.log({"Validation Dice Score": dice_score2,"Val_NSD": nsd2, "Epoch":epoch})  # 记录到 wandb
                pbar = tqdm(val_loader)
                for step, batch in enumerate(pbar):
                    image = batch["image"]
                    gt2D = batch["gt2D"]
                    boxes = batch["bboxes"]
                    image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
                    logits_pred, iou_pred = medsam_lite_model(image, boxes)
                    l_seg = seg_loss(logits_pred, gt2D)
                    
                    # l_ce = ce_loss(logits_pred, gt2D.float())
                    # mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
                    iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
                    l_iou = iou_loss(iou_pred, iou_gt)
                    # loss = mask_loss + iou_loss_weight * l_iou
                    # val_loss+=loss.item() 
                    l_seg = seg_loss(logits_pred, gt2D)
                    l_focal = focal_loss(logits_pred, gt2D)*5
                    loss=l_seg #+l_ce #+l_focal
                    val_loss+=loss.item()

                    iou_lo[step]=l_iou.item()
                    # ce_los[step]=l_ce.item()
                    seg_lo[step]=l_seg.item()
                    # focal_lo[step]=l_focal.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
            wandb.log({
                    "Val Loss": avg_val_loss,
                    "Epoch":epoch
                })
            
            wandb.log({
                "val iou_Loss": sum(iou_lo)/len(iou_lo),
                # "val ce_loss": sum(ce_los)/len(ce_los),
                "val seg_loss": sum(seg_lo)/len(seg_lo),
                # "val focal_loss": sum(focal_lo)/len(focal_lo),
                "Epoch":epoch
            })
            medsam_lite_model.train()  # 重新切换回训练模式
        if avg_val_loss*0.99 < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(checkpoint, join(work_dir, "medsam_lite_best_val.pth"))
        else:
            epochs_without_improvement += 1
        if dice_score2>dice_score:
            dice_score=dice_score2
            torch.save(checkpoint, join(work_dir, "medsam_lite_best_dice.pth"))
            
        wandb.log({
                "epochs without improvement":epochs_without_improvement ,
                "Epoch":epoch
            })
        
        # 如果验证集损失在连续10个epoch内没有改善，停止训练
        if epochs_without_improvement >= patience:
            print(f"stop training in epoch{epoch+1}")
            break