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

from dataset import CascadeMedSAMDataset
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
    "-resume", type=str, default='workdir_FL_auto/medsam_lite_best.pth',
    help="Path to the checkpoint to continue training."
)
parser.add_argument(
    "-work_dir", type=str, default="./workdir_FL_auto",
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
    "-bbox_shift", type=int, default=10,
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
parser.add_argument(
    "-json_prompt_path", type=str, required=True,
    help="Path to the JSON file containing TP/FP prompts."
)
parser.add_argument(
    "-val_json_prompt_path", type=str, default=None,
    help="Path to the JSON file containing TP/FP prompts for the Validation set."
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
    json_prompt_path = args.json_prompt_path
    val_json_prompt_path = args.val_json_prompt_path

    makedirs(work_dir, exist_ok=True)

    import wandb
    os.environ["WANDB_MODE"] = "online"
    #swanlab.init(project="MedSAM_sarcoma_box", name="lymphoma_train_medsam_box0",config={
    #swanlabab.init(project="MedSAM_sarcoma_box", name="lymphoma_train_fromscratch",config={
    #swanlab.init(project="MedSAM_sarcoma_box", name="2fold_sarcoma_pretrain_fromscratch1",config={
    wandb.init(project="DLBCL2FL", name="FL_Cascade",config={
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
    class ValNpzDataset(Dataset):
        def __init__(self, npz_dir, image_size=256, bbox_shift=5):
            self.npz_dir = npz_dir
            self.image_size = image_size
            self.target_length = image_size
            self.bbox_shift = bbox_shift
            self.samples = []
            if not os.path.isdir(npz_dir):
                raise FileNotFoundError(f"Validation npz directory not found: {npz_dir}")
            for f in sorted(os.listdir(npz_dir)):
                if f.endswith('.npz'):
                    with np.load(os.path.join(npz_dir, f), allow_pickle=True) as npz_data:
                        for z in range(npz_data['imgs'].shape[0]):
                            if np.any(npz_data['gts'][z] > 0):
                                self.samples.append((f, z))
            if not self.samples:
                raise RuntimeError(f"No positive validation slices were found in: {npz_dir}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            filename, z_index = self.samples[index]
            npz_path = os.path.join(self.npz_dir, filename)
            with np.load(npz_path, allow_pickle=True) as npz_data:
                img_2d = npz_data["imgs"][z_index].astype(np.float32)
                gt_2d = (npz_data["gts"][z_index] > 0).astype(np.uint8)

            img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            img_resize = resize_longest_side(img_3c, self.target_length).astype(np.float32)
            img_resize = (img_resize - img_resize.min()) / np.clip(
                img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None
            )
            img_padded = pad_image(img_resize, self.image_size)

            gt_resize = cv2.resize(
                gt_2d.astype(np.uint8),
                (img_resize.shape[1], img_resize.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            gt_padded = pad_image(gt_resize, self.image_size)

            y_indices, x_indices = np.where(gt_2d > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            scale = self.target_length / float(max(gt_2d.shape[0], gt_2d.shape[1]))
            resized_h, resized_w = img_resize.shape[:2]
            x_min = max(0, int(x_min * scale + 0.5) - random.randint(0, self.bbox_shift))
            x_max = min(resized_w - 1, int(x_max * scale + 0.5) + random.randint(0, self.bbox_shift))
            y_min = max(0, int(y_min * scale + 0.5) - random.randint(0, self.bbox_shift))
            y_max = min(resized_h - 1, int(y_max * scale + 0.5) + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

            return {
                "image": torch.tensor(np.transpose(img_padded, (2, 0, 1))).float(),
                "gt2D": torch.tensor(gt_padded[None, :, :]).float(),
                "bboxes": torch.tensor(bboxes).float(),
                "image_name": f"{filename}:{z_index}",
                "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
                "original_size": torch.tensor(np.array([img_2d.shape[0], img_2d.shape[1]])).long()
            }


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

    def get_bbox_from_gt2D(gt2D, bbox_shift=10):
        """
        直接从验证集的二维金标准中提取bbox（Oracle Validation），彻底摆脱外部 npy 框文件
        """
        y_indices, x_indices = np.where(gt2D > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return np.array([0, 0, 255, 255], dtype=np.float32)

        height, width = gt2D.shape
        x_min = max(0, int(np.min(x_indices)) - random.randint(0, bbox_shift))
        x_max = min(width - 1, int(np.max(x_indices)) + random.randint(0, bbox_shift))
        y_min = max(0, int(np.min(y_indices)) - random.randint(0, bbox_shift))
        y_max = min(height - 1, int(np.max(y_indices)) + random.randint(0, bbox_shift))
        
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)



    def MedSAM_infer_npz(gt_path_files, bbox_root):
        all_dices=0
        all_nsds=0
        item=0
        for filename in os.listdir(gt_path_files):
            if filename.endswith('.npz'):
                gt_path_file = os.path.join(gt_path_files, filename)
                #print(gt_path_file)
                npz_name = os.path.splitext(filename)[0]  # 提取npz文件名（不含后缀），如 "case_001"
                npz_data = np.load(gt_path_file, 'r', allow_pickle=True) # (H, W, 3)
                img_3D = npz_data['imgs'] # (Num, H, W)
                gt_3D = npz_data['gts'] # (Num, H, W)
                spacing = npz_data['spacing']
                seg_3D = np.zeros_like(gt_3D, dtype=np.uint8) # (Num, H, W)
                box_list = [dict() for _ in range(img_3D.shape[0])]

                for i in range(img_3D.shape[0]):

                    img_name = f"{npz_name}-{i:03d}.npy"
                    
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
                            box = get_bbox_from_gt2D(
                                gt2D=gt2D_padded,
                                bbox_shift=bbox_shift  # 原有参数
                            )
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

    def MedSAM_pipeline_infer_npz(npz_dir, json_prompt_path, medsam_model, device):
        import json

        all_dices = 0.0
        all_nsds = 0.0
        item = 0

        with open(json_prompt_path, "r", encoding="utf-8") as f:
            prompts_dict = json.load(f)

        for filename in os.listdir(npz_dir):
            if not filename.endswith(".npz"):
                continue

            npz_path = os.path.join(npz_dir, filename)
            case_id = os.path.splitext(filename)[0]
            with np.load(npz_path, "r", allow_pickle=True) as npz_data:
                img_3D = npz_data["imgs"]
                gt_3D = npz_data["gts"]
                if "spacing" in npz_data:
                    spacing = npz_data["spacing"]
                else:
                    spacing = np.array([4.0, 4.0, 4.0], dtype=np.float32)

            seg_3D = np.zeros_like(gt_3D, dtype=np.uint8)

            if case_id in prompts_dict:
                case_prompts = prompts_dict[case_id]
                z_boxes = {}
                for ptype in ["TP", "FP"]:
                    for prompt in case_prompts.get(ptype, []):
                        z = int(prompt["z"])
                        z_boxes.setdefault(z, []).append(prompt["box_2d"])

                for z, boxes in z_boxes.items():
                    if z < 0 or z >= img_3D.shape[0]:
                        continue

                    img_2d = img_3D[z]
                    H, W = img_2d.shape[:2]
                    img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)

                    img_256 = resize_longest_side(img_3c, 256)
                    newh, neww = img_256.shape[:2]
                    img_256 = (img_256 - img_256.min()) / np.clip(
                        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
                    )
                    img_256_padded = pad_image(img_256, 256)
                    img_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_embedding = medsam_model.image_encoder(img_tensor)

                    slice_seg = np.zeros((H, W), dtype=np.uint8)
                    scale = 256.0 / max(H, W)
                    for box in boxes:
                        y_min, x_min, y_max, x_max = box
                        box_256 = np.array(
                            [x_min * scale, y_min * scale, x_max * scale, y_max * scale],
                            dtype=np.float32,
                        )
                        sam_mask = medsam_inference(medsam_model, image_embedding, box_256, (newh, neww), (H, W))
                        slice_seg[sam_mask > 0] = 1

                    seg_3D[z] = slice_seg

            gt_binary = (gt_3D > 0).astype(np.uint8)
            seg_binary = (seg_3D > 0).astype(np.uint8)
            dsc = compute_dice_coefficient(gt_binary, seg_binary)
            surface_distance = compute_surface_distances(gt_binary, seg_binary, spacing_mm=spacing)
            nsd = compute_surface_dice_at_tolerance(surface_distance, 4.0)

            all_dices += dsc
            all_nsds += nsd
            item += 1

        if item == 0:
            return 0.0, 0.0
        return all_dices / item, all_nsds / item




    #%% sanity test of dataset class
    if do_sancheck:
        tr_dataset = CascadeMedSAMDataset(
            npz_dir=train_pathfile,
            json_prompt_path=json_prompt_path,
            image_size=256,
            bbox_shift=bbox_shift
        )
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


    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    #seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=False, reduction='mean')
    # weights = torch.tensor([1, 10])
    # ce_loss = nn.BCEWithLogitsLoss(reduction='mean',weight=weights)
    # iou_loss = nn.MSELoss(reduction='mean')
    focal_loss = monai.losses.FocalLoss(
        gamma=5.0, 
        alpha=0.75, 
        reduction='mean',
        to_onehot_y=False,  # 因为您的 GT 已经是 0/1 的二值 mask，不需要转 one-hot
        use_softmax=False,  # 只有两类且输出是一个通道，不要用 Softmax
    )
    # %%
    train_dataset = CascadeMedSAMDataset(
        npz_dir=args.train_pathfile,
        json_prompt_path=args.json_prompt_path,
        image_size=256,
        bbox_shift=args.bbox_shift
    )
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers, 
                              pin_memory=True)
    val_dataset = ValNpzDataset(npz_dir=args.test_pathfile, image_size=256, bbox_shift=args.bbox_shift)
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
    train_bbox_root = join(args.data_root, "bboxes")
    dice_score1,nsd1=MedSAM_infer_npz(train_pathfile, bbox_root=train_bbox_root)
    print(f"Epoch {0}: Oracle Train Dice = {dice_score1:.4f}, Oracle Train NSD = {nsd1:.4f}")
    wandb.log({"Oracle Train Dice": dice_score1,"Oracle Train NSD": nsd1,"Epoch":0})  # 记录到 wandb
    val_bbox_root = join(args.test_data_root, "bboxes")  # test_data_root是验证集npy根路径
    dice_score,nsd2=MedSAM_infer_npz(test_pathfile, bbox_root=val_bbox_root)
    print(f"Epoch {0}: Oracle Val Dice = {dice_score:.4f}, Oracle Val NSD = {nsd2:.4f}")
    wandb.log({"Oracle Val Dice": dice_score,"Oracle Val NSD": nsd2, "Epoch":0})  # 记录到 wandb
    if val_json_prompt_path is not None and os.path.isfile(val_json_prompt_path):
        pipeline_dice, pipeline_nsd = MedSAM_pipeline_infer_npz(
            test_pathfile, val_json_prompt_path, medsam_lite_model, device
        )
        print(f"Epoch {0}: Pipeline Val Dice = {pipeline_dice:.4f}, Pipeline Val NSD = {pipeline_nsd:.4f}")
        wandb.log({"Pipeline Val Dice": pipeline_dice, "Pipeline Val NSD": pipeline_nsd, "Epoch": 0})
        dice_score = pipeline_dice
    medsam_lite_model.train()
    iou=[]
    ce=[]
    seg=[]
    epochs_without_improvement=0
    for epoch in range(start_epoch + 1, num_epochs):
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        # iou_los = [1e10 for _ in range(len(train_loader))]
        # ce_los = [1e10 for _ in range(len(train_loader))]
        seg_los = [1e10 for _ in range(len(train_loader))]
        focal_los=[1e10 for _ in range(len(train_loader))]
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
            # iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            # l_iou = iou_loss(iou_pred, iou_gt)
            #loss = mask_loss + iou_loss_weight * l_iou
            #epoch_loss[step] = loss.item()
            l_seg = seg_loss(logits_pred, gt2D)
            l_focal = focal_loss(logits_pred, gt2D.float()) 
            loss=0.5*l_seg + 0.5*l_focal #+l_ce
            epoch_loss[step]=loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
            #iou_los[step]=l_iou.item()
            # ce_los[step]=l_ce.item()
            seg_los[step]=l_seg.item()
            focal_los[step]=l_focal.item()

            
        epoch_end_time = time()
        epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
        train_losses.append(epoch_loss_reduced)

        # lr_scheduler.step()
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
            "Train focal_loss": sum(focal_los)/len(focal_los),
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
            # iou_lo = [1e10 for _ in range(len(val_loader))]
            # ce_los = [1e10 for _ in range(len(val_loader))]
            seg_lo = [1e10 for _ in range(len(val_loader))]
            focal_lo=[1e10 for _ in range(len(val_loader))]
            val_loss = 0.0
            with torch.no_grad():
                if train_pathfile!=None:
                    dice_score1,nsd1=MedSAM_infer_npz(train_pathfile, bbox_root=train_bbox_root)
                    print(f"Epoch {epoch + 1}: Oracle Train Dice = {dice_score1:.4f}, Oracle Train NSD = {nsd1:.4f}")
                    wandb.log({"Oracle Train Dice": dice_score1,"Oracle Train NSD": nsd1, "Epoch":epoch})  # 记录到 wandb
                dice_score2, nsd2=MedSAM_infer_npz(test_pathfile, bbox_root=val_bbox_root)
                print(f"Epoch {epoch + 1}: Oracle Val Dice = {dice_score2:.4f}, Oracle Val NSD = {nsd2:.4f}")
                wandb.log({"Oracle Val Dice": dice_score2,"Oracle Val NSD": nsd2, "Epoch":epoch})  # 记录到 wandb
                val_dice_for_best = dice_score2
                if val_json_prompt_path is not None and os.path.isfile(val_json_prompt_path):
                    pipeline_dice, pipeline_nsd = MedSAM_pipeline_infer_npz(
                        test_pathfile, val_json_prompt_path, medsam_lite_model, device
                    )
                    print(
                        f"Epoch {epoch + 1}: Pipeline Val Dice = {pipeline_dice:.4f}, "
                        f"Pipeline Val NSD = {pipeline_nsd:.4f}"
                    )
                    wandb.log({"Pipeline Val Dice": pipeline_dice, "Pipeline Val NSD": pipeline_nsd, "Epoch": epoch})
                    val_dice_for_best = pipeline_dice
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
                    # iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
                    # l_iou = iou_loss(iou_pred, iou_gt)
                    # loss = mask_loss + iou_loss_weight * l_iou
                    # val_loss+=loss.item() 
                    l_seg = seg_loss(logits_pred, gt2D)
                    l_focal = focal_loss(logits_pred, gt2D.float()) 
                    loss=0.5*l_seg + 0.5*l_focal#+l_ce
                    val_loss+=loss.item()

                    # iou_lo[step]=l_iou.item()
                    # ce_los[step]=l_ce.item()
                    seg_lo[step]=l_seg.item()
                    focal_lo[step]=l_focal.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
            wandb.log({
                    "Val Loss": avg_val_loss,
                    "Epoch":epoch
                })
            
            wandb.log({
                # "val iou_Loss": sum(iou_lo)/len(iou_lo),
                # "val ce_loss": sum(ce_los)/len(ce_los),
                "val seg_loss": sum(seg_lo)/len(seg_lo),
                "val focal_loss": sum(focal_lo)/len(focal_lo),
                "Epoch":epoch
            })

            lr_scheduler.step(avg_val_loss)
            medsam_lite_model.train()  # 重新切换回训练模式
        if avg_val_loss*0.99 < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(checkpoint, join(work_dir, "medsam_lite_best_val.pth"))
        else:
            epochs_without_improvement += 1
        if val_dice_for_best > dice_score:
            dice_score = val_dice_for_best
            torch.save(checkpoint, join(work_dir, "medsam_lite_best_dice.pth"))
            
        wandb.log({
                "epochs without improvement":epochs_without_improvement ,
                "Epoch":epoch
            })
        
        # 如果验证集损失在连续10个epoch内没有改善，停止训练
        if epochs_without_improvement >= patience:
            print(f"stop training in epoch{epoch+1}")
            break
        
