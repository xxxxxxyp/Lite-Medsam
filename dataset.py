import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def resize_longest_side(image, target_length=256):
    """
    Expects a numpy array with shape HxWxC in uint8/float format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    return cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)


def pad_image(image, target_size=256):
    """
    Expects a numpy array with shape HxWxC or HxW in uint8/float format.
    """
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if padh < 0 or padw < 0:
        raise ValueError(f"Target size {target_size} is smaller than input shape {(h, w)}")
    if len(image.shape) == 3:
        return np.pad(image, ((0, padh), (0, padw), (0, 0)))
    return np.pad(image, ((0, padh), (0, padw)))


def _clip_box_xyxy(box, height, width):
    x_min, y_min, x_max, y_max = box
    x_min = int(np.clip(x_min, 0, max(width - 1, 0)))
    y_min = int(np.clip(y_min, 0, max(height - 1, 0)))
    x_max = int(np.clip(x_max, x_min, max(width - 1, 0)))
    y_max = int(np.clip(y_max, y_min, max(height - 1, 0)))
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def _gt_to_box_xyxy(gt_2d, bbox_shift=5):
    y_indices, x_indices = np.where(gt_2d > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Cannot extract bbox from an empty ground-truth mask.")

    height, width = gt_2d.shape
    x_min = max(0, int(np.min(x_indices)) - random.randint(0, bbox_shift))
    x_max = min(width - 1, int(np.max(x_indices)) + random.randint(0, bbox_shift))
    y_min = max(0, int(np.min(y_indices)) - random.randint(0, bbox_shift))
    y_max = min(height - 1, int(np.max(y_indices)) + random.randint(0, bbox_shift))
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


class CascadeMedSAMDataset(Dataset):
    def __init__(self, npz_dir, json_prompt_path, image_size=256, bbox_shift=5):
        self.npz_dir = npz_dir
        self.json_prompt_path = json_prompt_path
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift

        if not os.path.isdir(self.npz_dir):
            raise FileNotFoundError(f"npz directory not found: {self.npz_dir}")
        if not os.path.isfile(self.json_prompt_path):
            raise FileNotFoundError(f"prompt json not found: {self.json_prompt_path}")

        with open(self.json_prompt_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)

        self.prompt_index = {}
        self.samples = []
        for case_id, case_prompts in prompt_data.items():
            npz_path = os.path.join(self.npz_dir, f"{case_id}.npz")
            if not os.path.isfile(npz_path):
                continue

            slice_prompt_map = defaultdict(lambda: {"TP": [], "FP": []})
            for prompt_type in ("TP", "FP"):
                for item in case_prompts.get(prompt_type, []):
                    if "z" not in item or "box_2d" not in item:
                        continue
                    z_index = int(item["z"])
                    slice_prompt_map[z_index][prompt_type].append(item["box_2d"])

            if not slice_prompt_map:
                continue

            self.prompt_index[case_id] = dict(slice_prompt_map)
            for z_index in sorted(slice_prompt_map.keys()):
                self.samples.append({"case_id": case_id, "z": z_index})

        if not self.samples:
            raise RuntimeError(
                "No valid CascadeMedSAMDataset samples were built from the provided npz directory and prompt json."
            )

    def __len__(self):
        return len(self.samples)

    def _load_case_slice(self, case_id, z_index):
        npz_path = os.path.join(self.npz_dir, f"{case_id}.npz")
        with np.load(npz_path, allow_pickle=True) as npz_data:
            imgs = npz_data["imgs"]
            gts = npz_data["gts"]
            if z_index < 0 or z_index >= imgs.shape[0]:
                raise IndexError(f"Slice index {z_index} out of range for case {case_id}")

            img_2d = imgs[z_index].astype(np.float32)
            gt_2d = (gts[z_index] > 0).astype(np.uint8)
        return img_2d, gt_2d

    def _prepare_image(self, img_2d):
        img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
        img_resize = resize_longest_side(img_3c, self.target_length).astype(np.float32)
        img_resize = (img_resize - img_resize.min()) / np.clip(
            img_resize.max() - img_resize.min(),
            a_min=1e-8,
            a_max=None,
        )
        img_padded = pad_image(img_resize, self.image_size)
        img_tensor = torch.tensor(np.transpose(img_padded, (2, 0, 1))).float()
        return img_tensor, img_resize.shape[:2]

    def _prepare_mask(self, mask_2d, resized_hw):
        resized_h, resized_w = resized_hw
        if mask_2d.shape != (resized_h, resized_w):
            mask_resize = cv2.resize(
                mask_2d.astype(np.uint8),
                (resized_w, resized_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        else:
            mask_resize = mask_2d.astype(np.uint8)
        mask_padded = pad_image(mask_resize, self.image_size)
        return torch.tensor(mask_padded[None, :, :]).float()

    def _scale_box_to_256(self, box_yxyx, original_hw):
        original_h, original_w = original_hw
        scale = self.target_length / float(max(original_h, original_w))
        y_min, x_min, y_max, x_max = box_yxyx
        box_xyxy = np.array(
            [x_min * scale, y_min * scale, x_max * scale, y_max * scale],
            dtype=np.float32,
        )
        resized_h = int(original_h * scale + 0.5)
        resized_w = int(original_w * scale + 0.5)
        return _clip_box_xyxy(box_xyxy, resized_h, resized_w)

    def __getitem__(self, index):
        sample = self.samples[index]
        case_id = sample["case_id"]
        z_index = sample["z"]
        img_2d, gt_2d = self._load_case_slice(case_id, z_index)
        prompt_info = self.prompt_index[case_id][z_index]
        tp_boxes = prompt_info.get("TP", [])
        fp_boxes = prompt_info.get("FP", [])

        # 先按照约定概率决定优先尝试的采样策略：
        # A(25%): 使用前端假阳性框，强制把监督 mask 置零，让模型学会抑制 FP。
        # B(25%): 使用前端真阳性粗框，让模型学习在粗框内细化真实边界。
        # C(50%): 完全基于金标准 mask 重新生成扰动框，维持基础分割能力。
        rand_val = random.random()
        if rand_val < 0.25:
            selected_strategy = "A"
        elif rand_val < 0.5:
            selected_strategy = "B"
        else:
            selected_strategy = "C"

        # 若 A/B 所需提示框缺失，则按需求优先回退到 C。
        # 但在真实训练数据中，某些仅含 FP 的切片其 gt 可能本身为空，此时 C 无法构造金标准框，
        # 因此再进一步回退到当前切片可用的提示类型，保证数据集健壮性且不会返回非法框。
        if selected_strategy == "A" and not fp_boxes:
            selected_strategy = "C"
        elif selected_strategy == "B" and not tp_boxes:
            selected_strategy = "C"

        if selected_strategy == "C" and not np.any(gt_2d):
            if tp_boxes:
                selected_strategy = "B"
            elif fp_boxes:
                selected_strategy = "A"
            else:
                raise RuntimeError(f"Slice {case_id}:{z_index} has neither usable prompts nor positive gt.")

        if selected_strategy == "A":
            chosen_box = random.choice(fp_boxes)
            target_mask_2d = np.zeros_like(gt_2d, dtype=np.uint8)
            box_256 = self._scale_box_to_256(chosen_box, gt_2d.shape)
        elif selected_strategy == "B":
            chosen_box = random.choice(tp_boxes)
            target_mask_2d = gt_2d.astype(np.uint8)
            box_256 = self._scale_box_to_256(chosen_box, gt_2d.shape)
        else:
            target_mask_2d = gt_2d.astype(np.uint8)
            gt_box_xyxy = _gt_to_box_xyxy(gt_2d, self.bbox_shift)
            chosen_box = np.array(
                [gt_box_xyxy[1], gt_box_xyxy[0], gt_box_xyxy[3], gt_box_xyxy[2]],
                dtype=np.float32,
            )
            box_256 = self._scale_box_to_256(chosen_box, gt_2d.shape)

        image_tensor, resized_hw = self._prepare_image(img_2d)
        mask_tensor = self._prepare_mask(target_mask_2d, resized_hw)
        box_tensor = torch.tensor(box_256).float()

        if selected_strategy == "A" and torch.count_nonzero(mask_tensor).item() != 0:
            raise RuntimeError("FP branch mask tensor must be all zeros after preprocessing.")

        return {
            "image": image_tensor,
            "gt2D": mask_tensor,
            "bboxes": box_tensor,
            "image_name": f"{case_id}.npz:{z_index}",
            "new_size": torch.tensor(np.array(resized_hw)).long(),
            "original_size": torch.tensor(np.array(img_2d.shape)).long(),
        }
