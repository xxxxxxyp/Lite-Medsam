import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from os.path import join, basename, isfile, exists
from glob import glob
import argparse

# -------------------------- 1. 模型：features + gate + pos encoding + bbox/pres heads --------------------------
class LightweightLesionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=False),
        )
        # gate / attention map (same spatial resolution as final feature map)
        self.gate_conv = nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.gate_act = nn.Sigmoid()
        # global pooling and pos encoding
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(0.1, inplace=False)
        )
        # shared head
        self.head_fc = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout(0.3)
        )
        # bbox head: predict cx,cy,w,h (normalized 0-1)
        self.bbox_fc = nn.Linear(64, 4)
        self.bbox_act = nn.Sigmoid()
        # presence head
        self.pres_fc = nn.Linear(64, 1)
        self.pres_act = nn.Sigmoid()

        # Initialize bbox_fc and pres_fc biases later from dataset mean
        # (we'll do that in train_detector after computing dataset stats)

    def forward(self, x):
        features = self.features(x)  # (B,64,Hf,Wf)
        B, C, Hf, Wf = features.shape

        # gate (attention map)
        gate_logits = self.gate_conv(features)  # (B,1,Hf,Wf)
        gate = self.gate_act(gate_logits)       # sigmoid -> (B,1,Hf,Wf)

        # weight features by gate
        features_att = features * gate  # elementwise

        # global pooling
        gap_feat = self.global_max_pool(features_att).flatten(1)  # (B,64)

        # position encoding computed from weighted features (keeps some spatial signal)
        # grid in [0,1]
        x_grid = torch.linspace(0, 1, Wf, device=x.device).repeat(B, Hf, 1)  # (B,Hf,Wf)
        y_grid = torch.linspace(0, 1, Hf, device=x.device).repeat(B, Wf, 1).transpose(1, 2)  # (B,Hf,Wf)
        feat_weight = features_att.sum(dim=1)  # (B,Hf,Wf)
        feat_weight_sum = feat_weight.sum(dim=(1,2)) + 1e-6
        x_center = (feat_weight * x_grid).sum(dim=(1,2)) / feat_weight_sum
        y_center = (feat_weight * y_grid).sum(dim=(1,2)) / feat_weight_sum
        pos_feat_raw = torch.cat([x_center.unsqueeze(1), y_center.unsqueeze(1)], dim=1)  # (B,2)
        pos_feat = self.pos_encoder(pos_feat_raw)  # (B,32)

        # shared final feature
        final_feat = torch.cat([gap_feat, pos_feat], dim=1)  # (B,96)
        shared = self.head_fc(final_feat)  # (B,64)

        # heads
        bbox_params = self.bbox_act(self.bbox_fc(shared))  # (B,4) cx,cy,w,h in [0,1]
        pres = self.pres_act(self.pres_fc(shared)).squeeze(1)  # (B,)
        return bbox_params, pres, gate  # gate: (B,1,Hf,Wf)


# -------------------------- 2. Dataset（保留严格过滤 & 计算数据集均值用于初始化） --------------------------
class DetectorDataset(Dataset):
    def __init__(self, data_root, bbox_shift=5, data_aug=False, min_fg_pixels=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        assert exists(self.gt_path) and exists(self.img_path), "数据目录不存在"

        self.gt_files = sorted(glob(join(self.gt_path, '*.npy'), recursive=True))
        self.valid_files = []
        for f in self.gt_files:
            img_file = join(self.img_path, basename(f))
            if not isfile(img_file):
                continue
            try:
                gt = np.load(f, allow_pickle=True)
                fg_pixels = np.sum(gt != 0)
                if fg_pixels >= min_fg_pixels:
                    self.valid_files.append(f)
            except:
                continue
        assert len(self.valid_files) > 0, "无有效训练样本（需病灶像素>=min_fg_pixels）"
        self.gt_files = self.valid_files
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
        self.image_size = 256
        self.min_fg_pixels = min_fg_pixels

        # 计算数据集平均 bbox (cx,cy,w,h) 归一化，用于初始化 bbox head bias
        # 如果数据集很大，这一步会耗时；可以把统计结果缓存到文件后复用
        self.mean_bbox = self._compute_dataset_mean_bbox()

    def _compute_dataset_mean_bbox(self):
        cxs, cys, ws, hs = [], [], [], []
        for f in self.gt_files:
            gt = np.load(f, allow_pickle=True)
            # resize/pad to 256 -> reuse __getitem__ logic minimally
            # compute largest label bbox
            gt_resized = cv2.resize(gt, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            labels = np.unique(gt_resized)
            labels = labels[labels != 0]
            if len(labels) == 0:
                continue
            max_area = 0
            sel = labels[0]
            for lid in labels:
                area = np.sum(gt_resized == lid)
                if area > max_area:
                    max_area = area
                    sel = lid
            mask = (gt_resized == sel).astype(np.uint8)
            ys, xs = np.where(mask == 1)
            if len(xs) == 0:
                continue
            x1 = max(0, int(np.min(xs) - self.bbox_shift))
            x2 = min(self.image_size-1, int(np.max(xs) + self.bbox_shift))
            y1 = max(0, int(np.min(ys) - self.bbox_shift))
            y2 = min(self.image_size-1, int(np.max(ys) + self.bbox_shift))
            cx = (x1 + x2) / 2.0 / (self.image_size - 1)
            cy = (y1 + y2) / 2.0 / (self.image_size - 1)
            w = (x2 - x1 + 1) / (self.image_size - 1)
            h = (y2 - y1 + 1) / (self.image_size - 1)
            cxs.append(cx); cys.append(cy); ws.append(w); hs.append(h)
        if len(cxs) == 0:
            return np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float32)
        return np.array([np.mean(cxs), np.mean(cys), np.mean(ws), np.mean(hs)], dtype=np.float32)

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_files[index])
        img_3c = np.load(join(self.img_path, img_name), allow_pickle=True)
        gt = np.load(self.gt_files[index], allow_pickle=True)

        # image preprocess
        img_resize = self._resize_longest_side(img_3c)
        p99 = np.percentile(img_resize, 99)
        img_resize = np.clip(img_resize, 0, p99)
        img_resize = img_resize / (p99 + 1e-8)
        img_padded = self._pad_image(img_resize)
        img = np.transpose(img_padded, (2,0,1)).astype(np.float32)

        # gt preprocess
        gt_resize = cv2.resize(gt, (img_resize.shape[1], img_resize.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_padded = self._pad_image(gt_resize)

        unique_labels = np.unique(gt_padded)
        unique_labels = unique_labels[unique_labels != 0]
        if len(unique_labels) == 0:
            gt2D = np.zeros_like(gt_padded, dtype=np.uint8)
        else:
            max_area = 0
            selected_id = int(unique_labels[0])
            for lid in unique_labels:
                area = np.sum(gt_padded == lid)
                if area > max_area:
                    max_area = area
                    selected_id = int(lid)
            gt2D = np.uint8(gt_padded == selected_id)

        fg_pixels = np.sum(gt2D == 1)
        if fg_pixels < self.min_fg_pixels // 2 and len(unique_labels) > 0:
            areas = [np.sum(gt_padded == l) for l in unique_labels]
            largest_label = int(unique_labels[np.argmax(areas)])
            gt2D = np.uint8(gt_padded == largest_label)

        # bounding box (tight)
        y_fg, x_fg = np.where(gt2D == 1)
        if len(x_fg) == 0 or len(y_fg) == 0:
            center = self.image_size // 2
            half = 5
            bbox = np.array([center-half, center-half, center+half, center+half], dtype=np.float32)
        else:
            x_min = max(0, int(np.min(x_fg) - self.bbox_shift))
            x_max = min(self.image_size - 1, int(np.max(x_fg) + self.bbox_shift))
            y_min = max(0, int(np.min(y_fg) - self.bbox_shift))
            y_max = min(self.image_size - 1, int(np.max(y_fg) + self.bbox_shift))
            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        # data augmentation: add random translation to reduce center bias
        if self.data_aug:
            img, bbox = self._apply_data_aug_with_translation(img, bbox)

        bbox_norm = bbox / (self.image_size - 1)
        return {
            "image": torch.tensor(img),
            "bbox_norm": torch.tensor(bbox_norm).float(),
            "gt_mask": torch.tensor(gt2D).float()
        }

    def _resize_longest_side(self, image):
        long_side_length = self.image_size
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length / max(oldh, oldw)
        newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
        return cv2.resize(image, (neww, newh), interpolation=cv2.INTER_LINEAR)

    def _pad_image(self, image):
        h, w = image.shape[0], image.shape[1]
        padh = (self.image_size - h) // 2
        padw = (self.image_size - w) // 2
        padh1, padh2 = padh, self.image_size - h - padh
        padw1, padw2 = padw, self.image_size - w - padw
        if len(image.shape) == 3:
            return np.pad(image, ((padh1, padh2), (padw1, padw2), (0, 0)), mode='constant')
        else:
            return np.pad(image, ((padh1, padh2), (padw1, padw2)), mode='constant')

    def _apply_data_aug_with_translation(self, img, bbox):
        # original augmentations (scale, flip)
        H, W = self.image_size, self.image_size
        bbox = bbox.copy()
        # random small translation up to +/-16 pixels
        if random.random() > 0.5:
            max_shift = 16
            tx = random.randint(-max_shift, max_shift)
            ty = random.randint(-max_shift, max_shift)
            img_hwc = np.transpose(img, (1,2,0))
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img_hwc = cv2.warpAffine(img_hwc, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            img = np.transpose(img_hwc, (2,0,1))
            bbox = bbox + np.array([tx, ty, tx, ty], dtype=np.float32)
            # clip bbox
            bbox[0] = np.clip(bbox[0], 0, W-1)
            bbox[2] = np.clip(bbox[2], 0, W-1)
            bbox[1] = np.clip(bbox[1], 0, H-1)
            bbox[3] = np.clip(bbox[3], 0, H-1)
        # other augs preserved from prior implementation (scale/flip)
        # random scale (only shrink)
        if random.random() > 0.7:
            scale = random.uniform(0.85, 1.0)
            img_hwc = np.transpose(img, (1,2,0))
            new_h, new_w = int(H * scale), int(W * scale)
            img_hwc = cv2.resize(img_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padh = (H - new_h) // 2
            padw = (W - new_w) // 2
            img_hwc = np.pad(img_hwc, ((padh, H-new_h-padh), (padw, W-new_w-padw), (0,0)), mode='constant')
            img = np.transpose(img_hwc, (2,0,1))
            bbox = bbox * scale + np.array([padw, padh, padw, padh], dtype=np.float32)
        if random.random() > 0.5:
            img = np.flip(img, axis=-1).copy()
            bbox[0] = W - 1 - bbox[0]; bbox[2] = W - 1 - bbox[2]
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if random.random() > 0.5:
            img = np.flip(img, axis=-2).copy()
            bbox[1] = H - 1 - bbox[1]; bbox[3] = H - 1 - bbox[3]
            bbox[1], bbox[3] = bbox[3], bbox[1]
        return img, bbox


# -------------------------- 3. Loss: presence BCE + bbox L1/GIoU + area_pen + attention BCE --------------------------
def cxcywh_to_xyxy(box):
    cx, cy, w, h = box[...,0], box[...,1], box[...,2], box[...,3]
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return torch.stack([x1,y1,x2,y2], dim=-1)

def giou_loss(pred_boxes, target_boxes):
    eps = 1e-7
    px1, py1, px2, py2 = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]
    tx1, ty1, tx2, ty2 = target_boxes[:,0], target_boxes[:,1], target_boxes[:,2], target_boxes[:,3]
    ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
    iw = (ix2 - ix1).clamp(min=0.0); ih = (iy2 - iy1).clamp(min=0.0)
    inter = iw * ih
    area_p = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
    area_t = (tx2 - tx1).clamp(min=0.0) * (ty2 - ty1).clamp(min=0.0)
    union = area_p + area_t - inter + eps
    iou = inter / union
    ex1 = torch.min(px1, tx1); ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2); ey2 = torch.max(py2, ty2)
    ew = (ex2 - ex1).clamp(min=0.0); eh = (ey2 - ey1).clamp(min=0.0)
    enclose = ew * eh + eps
    giou = iou - (enclose - union) / enclose
    return giou

class CombinedLoss(nn.Module):
    def __init__(self, image_size=256, w_bce=1.0, w_l1=5.0, w_giou=2.0, w_area=1.0, w_att=1.0):
        super().__init__()
        self.image_size = image_size
        self.w_bce = w_bce
        self.w_l1 = w_l1
        self.w_giou = w_giou
        self.w_area = w_area
        self.w_att = w_att
        self.eps = 1e-6

    def forward(self, pred_params, pred_pres, target_gt, pred_gate):
        # pred_params: (B,4) cx,cy,w,h [0,1]
        # pred_pres: (B,)
        # target_gt: (B,H,W)
        # pred_gate: (B,1,Hf,Wf) in [0,1]
        device = pred_pres.device
        B = pred_pres.shape[0]

        # presence target
        gt_presence = (target_gt.sum(dim=(1,2)) > 0).float().to(device)

        # BCE for presence
        bce_pres = F.binary_cross_entropy(pred_pres, gt_presence)

        # compute gt boxes (normalized xyxy)
        gt_boxes = []
        for b in range(B):
            gt_mask = target_gt[b]
            nz = torch.nonzero(gt_mask == 1.0, as_tuple=False)
            if nz.shape[0] == 0:
                center = 0.5
                half = 5.0 / (self.image_size - 1)
                gt_boxes.append(torch.tensor([center-half, center-half, center+half, center+half], device=device))
            else:
                y = nz[:,0].float(); x = nz[:,1].float()
                x_min = torch.clamp(torch.min(x) / (self.image_size - 1), 0.0, 1.0)
                x_max = torch.clamp(torch.max(x) / (self.image_size - 1), 0.0, 1.0)
                y_min = torch.clamp(torch.min(y) / (self.image_size - 1), 0.0, 1.0)
                y_max = torch.clamp(torch.max(y) / (self.image_size - 1), 0.0, 1.0)
                gt_boxes.append(torch.stack([x_min, y_min, x_max, y_max]))
        gt_boxes = torch.stack(gt_boxes, dim=0)  # (B,4)

        # pred_params -> xyxy (safe, no in-place)
        min_wh = 1.0 / (self.image_size - 1)
        cx = pred_params[:,0]; cy = pred_params[:,1]
        w = pred_params[:,2].clamp(min=min_wh); h = pred_params[:,3].clamp(min=min_wh)
        pred_safe = torch.stack([cx, cy, w, h], dim=1)
        pred_xy = cxcywh_to_xyxy(pred_safe)
        pred_xy = torch.clamp(pred_xy, 0.0, 1.0)
        # ensure order
        x1 = torch.min(pred_xy[:,0], pred_xy[:,2])
        x2 = torch.max(pred_xy[:,0], pred_xy[:,2])
        y1 = torch.min(pred_xy[:,1], pred_xy[:,3])
        y2 = torch.max(pred_xy[:,1], pred_xy[:,3])
        pred_xy = torch.stack([x1,y1,x2,y2], dim=1)

        # L1 and GIoU for positive samples
        pos_mask = gt_presence > 0.5
        num_pos = int(pos_mask.sum().item())
        if num_pos > 0:
            pred_pos = pred_xy[pos_mask]
            gt_pos = gt_boxes[pos_mask]
            l1 = F.l1_loss(pred_pos, gt_pos, reduction='mean')
            giou = giou_loss(pred_pos, gt_pos)
            giou_loss_term = torch.mean(1.0 - giou)
        else:
            l1 = torch.tensor(0.0, device=device)
            giou_loss_term = torch.tensor(0.0, device=device)

        # area penalty: discourages large boxes for negatives; penalize huge boxes for positives
        pw = (pred_xy[:,2] - pred_xy[:,0]).clamp(min=0.0)
        ph = (pred_xy[:,3] - pred_xy[:,1]).clamp(min=0.0)
        area_ratio = pw * ph
        neg_mask = ~pos_mask
        neg_area_pen = area_ratio[neg_mask].mean() if neg_mask.sum().item() > 0 else torch.tensor(0.0, device=device)
        pos_area_pen = torch.clamp(area_ratio[pos_mask] - 0.6, min=0.0).mean() if pos_mask.sum().item() > 0 else torch.tensor(0.0, device=device)
        area_pen = neg_area_pen + pos_area_pen

        # attention/gate supervision: downsample target_gt to gate spatial size and compute BCE
        # pred_gate is (B,1,Hf,Wf)
        gate_size = pred_gate.shape[2:]
        # downsample using nearest (preserve mask)
        gt_mask_down = F.interpolate(target_gt.unsqueeze(1), size=gate_size, mode='nearest').squeeze(1)  # (B,Hf,Wf)
        gate_pred = pred_gate.squeeze(1)  # (B,Hf,Wf)
        att_loss = F.binary_cross_entropy(gate_pred, gt_mask_down)

        loss = self.w_bce * bce_pres + self.w_l1 * l1 + self.w_giou * giou_loss_term + self.w_area * area_pen + self.w_att * att_loss

        return loss, {
            "bce": float(bce_pres.item()),
            "l1": float(l1.item() if isinstance(l1, torch.Tensor) else l1),
            "giou": float(giou_loss_term.item() if isinstance(giou_loss_term, torch.Tensor) else giou_loss_term),
            "area_pen": float(area_pen.item() if isinstance(area_pen, torch.Tensor) else area_pen),
            "att": float(att_loss.item()),
            "num_pos": num_pos
        }


# -------------------------- 4. 工具：mask->bbox & IoU --------------------------
def mask_to_bbox(mask: torch.Tensor, image_size: int):
    nz = torch.nonzero(mask == 1.0, as_tuple=False)
    if nz.shape[0] == 0:
        center = image_size // 2
        half = 5
        return torch.tensor([center-half, center-half, center+half, center+half], dtype=torch.float32)
    y = nz[:,0].float(); x = nz[:,1].float()
    x_min = float(torch.clamp(torch.min(x), 0, image_size-1))
    x_max = float(torch.clamp(torch.max(x), 0, image_size-1))
    y_min = float(torch.clamp(torch.min(y), 0, image_size-1))
    y_max = float(torch.clamp(torch.max(y), 0, image_size-1))
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

def bbox_iou_tensor(boxA: torch.Tensor, boxB: torch.Tensor):
    xa1, ya1, xa2, ya2 = float(boxA[0]), float(boxA[1]), float(boxA[2]), float(boxA[3])
    xb1, yb1, xb2, yb2 = float(boxB[0]), float(boxB[1]), float(boxB[2]), float(boxB[3])
    xa1, xa2 = min(xa1, xa2), max(xa1, xa2)
    ya1, ya2 = min(ya1, ya2), max(ya1, ya2)
    xb1, xb2 = min(xb1, xb2), max(xb1, xb2)
    yb1, yb2 = min(yb1, yb2), max(yb1, yb2)
    inter_x1 = max(xa1, xb1); inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2); inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0); inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
    inter_area = inter_w * inter_h
    areaA = max(0.0, xa2 - xa1 + 1.0) * max(0.0, ya2 - ya1 + 1.0)
    areaB = max(0.0, xb2 - xb1 + 1.0) * max(0.0, yb2 - yb1 + 1.0)
    union = areaA + areaB - inter_area + 1e-8
    return inter_area / union


# -------------------------- 5. 训练流程：包括 bias 初始化 & 使用 attention loss --------------------------
def train_detector(
    train_data_root: str,
    val_data_root: str,
    work_dir: str,
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4,
    num_epochs: int = 50,
    lr: float = 5e-4
):
    print(f"[Detector] 加载训练集：{train_data_root}")
    train_dataset = DetectorDataset(train_data_root, bbox_shift=3, data_aug=True, min_fg_pixels=15)
    print(f"[Detector] 加载验证集：{val_data_root}")
    val_dataset = DetectorDataset(val_data_root, bbox_shift=3, data_aug=False, min_fg_pixels=15)
    print(f"[Detector] 有效训练样本数：{len(train_dataset)}, 有效验证样本数：{len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device(device)
    model = LightweightLesionDetector().to(device)

    # initialize bbox & pres bias from dataset statistics to avoid center bias from random init
    mean_bbox = train_dataset.mean_bbox  # numpy array [cx,cy,w,h]
    # clamp mean to avoid exact 0/1
    eps = 1e-3
    mean_bbox = np.clip(mean_bbox, eps, 1-eps)
    # set bias of linear layer before Sigmoid to logit(mean)
    def logit(p): return np.log(p/(1-p))
    with torch.no_grad():
        model.bbox_fc.bias.data = torch.tensor(logit(mean_bbox), dtype=torch.float32, device=device)
        # presence bias: set to positive (most samples are positive after filtering), but not extreme
        mean_pres = min(0.99, max(0.01, np.mean([1.0 for _ in range(len(train_dataset))])))  # mostly 1.0 because dataset filtered
        model.pres_fc.bias.data = torch.tensor(logit(mean_pres), dtype=torch.float32, device=device)

    criterion = CombinedLoss(image_size=256, w_bce=1.0, w_l1=5.0, w_giou=2.0, w_area=1.0, w_att=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    model_save_dir = join(work_dir, "lightweight_detector")
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = join(model_save_dir, "detector_best.pth")
    best_val_loss = float('inf')

    print(f"[Detector] 开始训练（设备：{device}，轮次：{num_epochs}）")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        stats = {"bce":0.0, "l1":0.0, "giou":0.0, "area_pen":0.0, "att":0.0, "num_pos":0}
        total_samples = 0
        iou_sum_train = 0.0
        iou_count_train = 0
        area_ratio_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
        for batch in pbar:
            images = batch["image"].to(device)
            gt_bbox_norm = batch["bbox_norm"].to(device)
            gt_mask = batch["gt_mask"].to(device)

            optimizer.zero_grad()
            pred_params, pred_pres, pred_gate = model(images)  # new: gate returned
            loss_tensor, loss_comp = criterion(pred_params, pred_pres, gt_mask, pred_gate)
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss_tensor.item() * images.size(0)
            stats["bce"] += loss_comp["bce"] * images.size(0)
            stats["l1"] += loss_comp["l1"] * images.size(0)
            stats["giou"] += loss_comp["giou"] * images.size(0)
            stats["area_pen"] += loss_comp["area_pen"] * images.size(0)
            stats["att"] += loss_comp["att"] * images.size(0)
            stats["num_pos"] += loss_comp["num_pos"]
            total_samples += images.size(0)

            pbar.set_postfix({"loss": loss_tensor.item()})

            # logging IoU and area
            with torch.no_grad():
                B = pred_params.shape[0]
                for i in range(B):
                    pr = pred_params[i].cpu()
                    # ensure w,h min
                    min_wh = 1.0/(256-1)
                    w = max(min_wh, float(pr[2].item()))
                    h = max(min_wh, float(pr[3].item()))
                    cx = float(pr[0].item()); cy = float(pr[1].item())
                    x1f = (cx - w/2.0)*(256-1); y1f = (cy - h/2.0)*(256-1)
                    x2f = (cx + w/2.0)*(256-1); y2f = (cy + h/2.0)*(256-1)
                    x1 = max(0.0, min(x1f,x2f)); x2 = min(255.0, max(x1f,x2f))
                    y1 = max(0.0, min(y1f,y2f)); y2 = min(255.0, max(y1f,y2f))
                    pred_box = torch.tensor([x1,y1,x2,y2], dtype=torch.float32)
                    gt_b = mask_to_bbox(gt_mask[i].cpu(), 256)
                    iou = bbox_iou_tensor(pred_box, gt_b)
                    iou_sum_train += iou
                    iou_count_train += 1
                    area_ratio_sum += ((x2-x1+1.0)*(y2-y1+1.0)) / (256.0*256.0)

        avg_train_loss = train_loss / max(1, total_samples)
        avg_bce = stats["bce"] / max(1, total_samples)
        avg_l1 = stats["l1"] / max(1, total_samples)
        avg_giou = stats["giou"] / max(1, total_samples)
        avg_area_pen = stats["area_pen"] / max(1, total_samples)
        avg_att = stats["att"] / max(1, total_samples)
        mean_iou_train = iou_sum_train / max(1, iou_count_train)
        mean_area_ratio_train = area_ratio_sum / max(1, iou_count_train)

        # validation
        model.eval()
        val_loss = 0.0
        val_stats = {"bce":0.0, "l1":0.0, "giou":0.0, "area_pen":0.0, "att":0.0, "num_pos":0}
        val_total = 0
        iou_sum_val = 0.0
        iou_count_val = 0
        area_ratio_sum_val = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)")
            for batch in pbar:
                images = batch["image"].to(device)
                gt_bbox_norm = batch["bbox_norm"].to(device)
                gt_mask = batch["gt_mask"].to(device)

                pred_params, pred_pres, pred_gate = model(images)
                loss_tensor, loss_comp = criterion(pred_params, pred_pres, gt_mask, pred_gate)
                val_loss += loss_tensor.item() * images.size(0)
                val_stats["bce"] += loss_comp["bce"] * images.size(0)
                val_stats["l1"] += loss_comp["l1"] * images.size(0)
                val_stats["giou"] += loss_comp["giou"] * images.size(0)
                val_stats["area_pen"] += loss_comp["area_pen"] * images.size(0)
                val_stats["att"] += loss_comp["att"] * images.size(0)
                val_stats["num_pos"] += loss_comp["num_pos"]
                val_total += images.size(0)

                # metrics
                B = pred_params.shape[0]
                for i in range(B):
                    pr = pred_params[i].cpu()
                    min_wh = 1.0/(256-1)
                    w = max(min_wh, float(pr[2].item()))
                    h = max(min_wh, float(pr[3].item()))
                    cx = float(pr[0].item()); cy = float(pr[1].item())
                    x1f = (cx - w/2.0)*(256-1); y1f = (cy - h/2.0)*(256-1)
                    x2f = (cx + w/2.0)*(256-1); y2f = (cy + h/2.0)*(256-1)
                    x1 = max(0.0, min(x1f,x2f)); x2 = min(255.0, max(x1f,x2f))
                    y1 = max(0.0, min(y1f,y2f)); y2 = min(255.0, max(y1f,y2f))
                    pred_box = torch.tensor([x1,y1,x2,y2], dtype=torch.float32)
                    gt_b = mask_to_bbox(gt_mask[i].cpu(), 256)
                    iou = bbox_iou_tensor(pred_box, gt_b)
                    iou_sum_val += iou
                    iou_count_val += 1
                    area_ratio_sum_val += ((x2-x1+1.0)*(y2-y1+1.0)) / (256.0*256.0)

        avg_val_loss = val_loss / max(1, val_total)
        avg_v_bce = val_stats["bce"] / max(1, val_total)
        avg_v_l1 = val_stats["l1"] / max(1, val_total)
        avg_v_giou = val_stats["giou"] / max(1, val_total)
        avg_v_area_pen = val_stats["area_pen"] / max(1, val_total)
        avg_v_att = val_stats["att"] / max(1, val_total)
        mean_iou_val = iou_sum_val / max(1, iou_count_val)
        mean_area_ratio_val = area_ratio_sum_val / max(1, iou_count_val)

        lr_scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }, model_save_path)
            print(f"[Detector] 保存最佳模型（验证损失：{best_val_loss:.6f}）")

        print(f"[Detector] Epoch {epoch+1} | 训练损失：{avg_train_loss:.6f} | 验证损失：{avg_val_loss:.6f} | 学习率：{optimizer.param_groups[0]['lr']:.6f}")
        print(f"[Detector] 组件 (Train) BCE:{avg_bce:.4f} L1:{avg_l1:.4f} GIOU:{avg_giou:.4f} AREApen:{avg_area_pen:.4f} ATT:{avg_att:.4f}")
        print(f"[Detector] 组件 (Val)   BCE:{avg_v_bce:.4f} L1:{avg_v_l1:.4f} GIOU:{avg_v_giou:.4f} AREApen:{avg_v_area_pen:.4f} ATT:{avg_v_att:.4f}")
        print(f"[Detector] 平均 IoU Train: {mean_iou_train:.4f}, Val: {mean_iou_val:.4f}")
        print(f"[Detector] 平均预测框面积比 Train: {mean_area_ratio_train:.4f}, Val: {mean_area_ratio_val:.4f}")
        print(f"[Detector] 正样本数 (Train total across batches): {stats['num_pos']}, (Val): {val_stats['num_pos']}")

    print(f"[Detector] 训练完成！最佳模型保存路径：{model_save_path}")
    return model_save_path


# -------------------------- 6. 推理接口（cx,cy,w,h -> x1,y1,x2,y2） --------------------------
def generate_lesion_bbox(image_tensor: torch.Tensor, detector_model: nn.Module, image_size: int = 256, min_size: int = 10) -> np.ndarray:
    if len(image_tensor.shape) == 4:
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)
        else:
            raise ValueError("不支持多Batch输入！")
    if image_tensor.shape == (image_size, image_size, 3):
        image_tensor = image_tensor.permute(2,0,1)
    elif image_tensor.shape != (3, image_size, image_size):
        raise ValueError(f"输入格式错误：{image_tensor.shape}")
    image_tensor = image_tensor.float()
    if image_tensor.min() < 0 or image_tensor.max() > 1:
        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    detector_model.eval()
    with torch.no_grad():
        pred_params, pres, gate = detector_model(image_tensor.unsqueeze(0))
        params = pred_params.squeeze(0).cpu()
        cx, cy, w, h = float(params[0].item()), float(params[1].item()), float(params[2].item()), float(params[3].item())
        min_wh = 1.0/(image_size-1)
        w = max(w, min_wh); h = max(h, min_wh)
        x1 = int(max(0, (cx - w/2.0)*(image_size-1)))
        y1 = int(max(0, (cy - h/2.0)*(image_size-1)))
        x2 = int(min(image_size-1, (cx + w/2.0)*(image_size-1)))
        y2 = int(min(image_size-1, (cy + h/2.0)*(image_size-1)))

        # postprocess
        if x2-x1 < min_size:
            x1 = max(0, x1 - (min_size - (x2-x1))//2)
            x2 = min(image_size-1, x2 + (min_size - (x2-x1+1))//2)
        if y2-y1 < min_size:
            y1 = max(0, y1 - (min_size - (y2-y1))//2)
            y2 = min(image_size-1, y2 + (min_size - (y2-y1+1))//2)
        if (x2-x1) > 0.9*image_size and (y2-y1) > 0.9*image_size:
            center_x = (x1+x2)//2; center_y = (y1+y2)//2
            new_size = max(min_size, int(0.5*image_size))
            x1 = max(0, center_x - new_size//2); x2 = min(image_size-1, center_x + new_size//2)
            y1 = max(0, center_y - new_size//2); y2 = min(image_size-1, center_y + new_size//2)
        if x1 >= x2 or y1 >= y2:
            center = image_size//2; half = min_size//2
            return np.array([center-half, center-half, center+half, center+half], dtype=np.int32)
        return np.array([x1,y1,x2,y2], dtype=np.int32)


def load_detector_model(model_path: str, device: str = "cuda") -> nn.Module:
    assert exists(model_path), f"模型文件不存在：{model_path}"
    device = torch.device(device)
    model = LightweightLesionDetector().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Detector] 成功加载模型：{model_path}（设备：{device}）")
    return model


# -------------------------- 7. 训练入口 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="轻量病灶检测器训练脚本")
    parser.add_argument("--train-data-root", default="/data/xyp/下载/data/npy/FL/train_0", help="训练集根目录")
    parser.add_argument("--val-data-root", default="/data/xyp/下载/data/npy/FL/val_0", help="验证集根目录")
    parser.add_argument("--work-dir", default="/data/xyp/下载/LiteMedSAM", help="模型保存目录")
    parser.add_argument("--device", default="cuda:2", help="训练设备")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--num-epochs", type=int, default=50, help="训练轮次")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_detector(
        train_data_root=args.train_data_root,
        val_data_root=args.val_data_root,
        work_dir=args.work_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        lr=args.lr
    )