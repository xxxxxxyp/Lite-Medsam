import os
import numpy as np
import torch
import cv2
from glob import glob
from os.path import join, exists, basename, isfile
from tqdm import tqdm

# 从 lightweight_detector 导入模型类与推理函数（不使用其 load_detector_model，使用兼容加载）
from lightweight_detector import LightweightLesionDetector, generate_lesion_bbox

# -------------------------- 配置参数（和主程序保持一致）--------------------------
detector_model_path = "/data/xyp/下载/LiteMedSAM/lightweight_detector/detector_best.pth"
detector_device = "cuda:2"  # 与主程序设备一致
image_size = 256  # 主程序训练图像尺寸
bbox_shift = 5  # 可选：保留原有的轻微扰动（和主程序一致）

# 要处理的数据集路径（训练集+验证集）
data_paths = {
    "train": "/data/xyp/下载/data/npy/FL/train_0",
    "val": "/data/xyp/下载/data/npy/FL/val_0"
}

# -------------------------- 复用主程序的图像预处理函数 --------------------------
def resize_longest_side(image, target_length=256):
    """和主程序NpyDataset中的resize_longest_side完全一致"""
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size=256):
    """和主程序NpyDataset中的pad_image完全一致"""
    h, w = image.shape[0], image.shape[1]
    # pad to bottom-right to reach target_size (matches previous pad logic used here)
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3:  # 图像（H,W,3）
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else:  # GT（H,W）
        image_padded = np.pad(image, ((0, padh), (0, padw)))
    return image_padded

# -------------------------- 兼容加载模型（跳过 shape 不匹配的参数） --------------------------
def load_detector_compat(model_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    兼容加载模型：当 checkpoint 与当前模型定义有参数名或形状不匹配时，
    跳过这些参数（保持当前模型初始化值），并打印被跳过的键，避免 load_state_dict 抛错。
    """
    assert os.path.exists(model_path), f"模型文件不存在：{model_path}"
    device_t = torch.device(device)
    ckpt = torch.load(model_path, map_location=device_t)
    # ckpt 可能直接是 state_dict，也可能是 dict 包含 "model_state_dict"
    ckpt_state = ckpt.get("model_state_dict", ckpt)
    # init model
    model = LightweightLesionDetector().to(device_t)
    model_state = model.state_dict()

    # Build a filtered state dict with only matching keys and shapes
    matched = {}
    skipped = []
    for k_ck, v_ck in ckpt_state.items():
        if k_ck in model_state:
            try:
                # some checkpoint tensors might be CPU numpy arrays or 0-d tensors; convert to tensor
                if not isinstance(v_ck, torch.Tensor):
                    v_ck = torch.tensor(v_ck)
                if v_ck.shape == model_state[k_ck].shape:
                    matched[k_ck] = v_ck.to(device_t)
                else:
                    skipped.append((k_ck, getattr(v_ck, "shape", type(v_ck)), model_state[k_ck].shape))
            except Exception as e:
                skipped.append((k_ck, f"error reading ckpt shape: {e}", model_state[k_ck].shape))
        else:
            skipped.append((k_ck, getattr(v_ck, "shape", type(v_ck)), None))

    # Load matched keys (non-strict)
    if len(matched) == 0:
        print("[load_detector_compat] WARNING: no matching parameters found to load from checkpoint.")
    missing_keys = set(model_state.keys()) - set(matched.keys())
    try:
        model.load_state_dict(matched, strict=False)
    except Exception as e:
        # Fallback: update model_state then load full dict (safe because matched keys have correct shapes)
        temp_state = model.state_dict()
        temp_state.update(matched)
        model.load_state_dict(temp_state)
    print(f"[load_detector_compat] Loaded {len(matched)} parameters into model. Skipped {len(skipped)} checkpoint keys.")
    if len(skipped) > 0:
        print("[load_detector_compat] Skipped keys (ckpt_key, ckpt_shape, model_shape):")
        for k, ck_shape, m_shape in skipped:
            print(f"  - {k}: ckpt_shape={ck_shape}, model_shape={m_shape}")
    model.eval()
    return model

# -------------------------- 生成并保存bbox --------------------------
def precompute_single_dataset(data_root):
    """处理单个数据集（train/val），生成所有图像的bbox"""
    # 数据路径（和主程序一致）
    img_path = join(data_root, "imgs")
    gt_path = join(data_root, "gts")
    bbox_save_path = join(data_root, "bboxes")  # 新增：保存bbox的文件夹
    os.makedirs(bbox_save_path, exist_ok=True)

    # 获取所有GT文件（按GT文件名对应图像）
    gt_files = sorted(glob(join(gt_path, '*.npy'), recursive=True))
    gt_files = [f for f in gt_files if isfile(join(img_path, basename(f)))]

    print(f"开始处理 {data_root}，共 {len(gt_files)} 张图像...")

    # 加载检测器（仅加载一次） —— 使用兼容加载函数
    detector_model = load_detector_compat(model_path=detector_model_path, device=detector_device)
    detector_model.eval()
    for param in detector_model.parameters():
        param.requires_grad = False

    # 遍历所有图像生成bbox
    for gt_file in tqdm(gt_files):
        img_name = basename(gt_file)  # 图像名和GT名一致
        img_path_full = join(img_path, img_name)
        bbox_save_full = join(bbox_save_path, img_name)  # bbox保存路径（.npy格式）

        # 跳过已生成的bbox
        if exists(bbox_save_full):
            continue

        # 1. 加载图像并做和主程序一致的预处理
        img_3c = np.load(img_path_full, 'r', allow_pickle=True)  # (H,W,3)
        img_resize = resize_longest_side(img_3c, target_length=image_size)
        # 和训练一致的 99%-clip 归一化会更接近训练时数据分布，但这里使用 min-max 归一化以稳定推理
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None)
        img_padded = pad_image(img_resize, target_size=image_size)  # (256,256,3)
        img_tensor = torch.tensor(img_padded).permute(2, 0, 1).unsqueeze(0).to(detector_device)  # (1,3,256,256)

        # 2. 加载GT（用于兜底：如果检测器生成的bbox太小） 
        gt = np.load(gt_file, 'r', allow_pickle=True)  # (H,W)
        gt_resize = cv2.resize(gt, (img_resize.shape[1], img_resize.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_padded = pad_image(gt_resize, target_size=image_size)  # (256,256)
        label_ids = np.unique(gt_padded)[1:]
        if len(label_ids) > 0:
            gt2D = np.uint8(gt_padded == label_ids[0])  # 取第一个病灶（预处理无需随机）
        else:
            gt2D = np.zeros((256, 256), dtype=np.uint8)

        # 3. 用检测器生成bbox（和主程序原逻辑一致）
        with torch.no_grad():
            lesion_bbox = generate_lesion_bbox(
                image_tensor=img_tensor,
                detector_model=detector_model,
                image_size=image_size
            )

        # 4. 兜底逻辑：如果bbox太小，用GT调整（和主程序一致）
        x_min, y_min, x_max, y_max = lesion_bbox
        if x_max - x_min < 5 or y_max - y_min < 5:
            if np.sum(gt2D) > 0:
                y_indices, x_indices = np.where(gt2D > 0)
                x_min = max(0, np.min(x_indices) - 3)
                x_max = min(255, np.max(x_indices) + 3)
                y_min = max(0, np.min(y_indices) - 3)
                y_max = min(255, np.max(y_indices) + 3)
                lesion_bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.int32)
            else:
                lesion_bbox = np.array([0, 0, 255, 255], dtype=np.int32)  # 全图框

        # 5. 保存bbox（x1,y1,x2,y2格式，和主程序要求一致）
        np.save(bbox_save_full, lesion_bbox)

    print(f"{data_root} 处理完成！bbox保存在 {bbox_save_path}")

if __name__ == "__main__":
    # 处理训练集和验证集
    for data_type, data_root in data_paths.items():
        precompute_single_dataset(data_root)