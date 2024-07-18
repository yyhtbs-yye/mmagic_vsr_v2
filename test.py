import torch
from mmagic.models.basicsr_utils.vidt_proc import window_partition
D = 6
H = 64
W = 64
device = 'cuda'
window_size = [6, 16, 16]
shift_size = [3, 8, 8]

img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
cnt = 0
for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
    for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
        for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
            img_mask[:, d, h, w, :] = cnt
            cnt += 1

a = 1
mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

