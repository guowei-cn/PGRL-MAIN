import os
import io
import random

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, ToPILImage
from lib.frequency import PoisonFre
from lib.rawDataProcessing import freq_img, blend_img

# -------------------------
# Your trigger function (kept as close as possible)
# -------------------------
def pattern_img(image, size_trigger=6):
    """
    image tensor with pixel value from [0, 1]
    :param image: image tensor with shape [channel, width, height]
                  (in practice we'll pass a PIL image and convert)
    :return: image (numpy array) with trigger
    """
    max, min = [1, 1, 1], [0, 0, 0]
    To_Tensor = ToTensor()
    image = To_Tensor(image)  # -> tensor [C, H, W] in [0, 1]

    # the trigger's width and height (dividable by 3)
    len_grid = size_trigger / 3
    mask = np.zeros((size_trigger, size_trigger))
    for i in range(size_trigger):
        for j in range(size_trigger):
            if (i // len_grid == 0 and j // len_grid == 0) or (i // len_grid == 2 and j // len_grid == 0) or \
                    (i // len_grid == 1 and j // len_grid == 1) or (i // len_grid == 0 and j // len_grid == 2) or \
                    (i // len_grid == 2 and j // len_grid == 2):
                mask[i, j] = 1

    right_down_corner = [5, 5]
    if len(image.shape) == 2:
        W, H = image.shape
        C = 1
        image = torch.unsqueeze(image, dim=0)
    else:
        C, W, H = image.shape

    for c in range(C):
        for i in range(size_trigger):
            for j in range(size_trigger):
                if mask[i][j] == 1:
                    image[c][W - (right_down_corner[0] + i)][H - (right_down_corner[1] + j)] = max[c]
                else:
                    image[c][W - (right_down_corner[0] + i)][H - (right_down_corner[1] + j)] = min[c]

    if C == 1:
        image = torch.squeeze(image, dim=0)

    To_PIL_image = ToPILImage()
    image = To_PIL_image(image)
    image = np.array(image)

    return image  # numpy array HxWxC


# -------------------------
# Helpers
# -------------------------
def ensure_folder(path):
    os.makedirs(path, exist_ok=True)


def encode_pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# -------------------------
# Export train split: 95% clean, 5% poisoned
# -------------------------


def export_train_clean_and_poison(
    dataset,
    attack_type,
    out_folder,
    poison_target,
    poison_ratio=0.05,
    trigger_size=6,
    seed=42,
):
    """
    dataset: CIFAR10(train=True, transform=None)
    attack_type: 'pattern' or 'freq'
    out_folder: where to write .bin and _index.txt
    poison_target: int, target class for clean-label attack (e.g. 0..9 for CIFAR-10)
    poison_ratio: fraction of *entire* train set to be poisoned (e.g. 0.05 -> 5%)
    """

    ensure_folder(out_folder)

    clean_bin_path = os.path.join(out_folder, "trainSet_clean_list.bin")
    clean_idx_path = os.path.join(out_folder, "trainSet_clean_list_index.txt")

    poison_bin_path = os.path.join(out_folder, "trainSet_poisoned_list.bin")
    poison_idx_path = os.path.join(out_folder, "trainSet_poisoned_list_index.txt")

    n = len(dataset)
    n_poison = int(poison_ratio * n)

    # ------------------------------------------------------------------
    # 1) Find all indices whose label == poison_target
    #    (clean-label: poisons come only from target class)
    # ------------------------------------------------------------------
    if attack_type == 'freq':
        target_indices = [i for i, (_, lab) in enumerate(dataset) if lab == poison_target]
        n_poison = len(target_indices)
        rng = random.Random(seed)
        poison_indices = set(rng.sample(target_indices, n_poison))
        cover_indices = []
    elif attack_type == 'pattern':
        rng = random.Random(seed)
        poison_indices = set(rng.sample(range(n), n_poison))
        cover_indices = []
    elif attack_type in ['adapblend', 'pattern_cover']:
        rng = random.Random(seed)
        poison_cover_indices = list(rng.sample(range(n), n_poison * 2))
        poison_indices = set(poison_cover_indices[:n_poison])
        cover_indices = set(poison_cover_indices[n_poison:])
    else:
        raise ValueError('No implementation')

    print(f"Total train samples: {n}")
    print(f"Poisoned train samples ({poison_ratio} of all): {n_poison}")
    print(f"Clean train samples: {n - n_poison}")
    print(f"Target class: {poison_target}, total target samples: {len(poison_indices)}")

    clean_offset = 0
    poison_offset = 0

    with open(clean_bin_path, "wb") as clean_bin_f, \
         open(clean_idx_path, "w") as clean_idx_f, \
         open(poison_bin_path, "wb") as poison_bin_f, \
         open(poison_idx_path, "w") as poison_idx_f:

        for i, (img, label) in enumerate(dataset):

            if i in poison_indices:
                # -------------------- POISONED (clean-label) --------------------
                # label is already poison_target by construction (since we only
                # selected indices with label == poison_target)
                if attack_type in ['pattern', 'pattern_cover']:
                    poisoned_np = pattern_img(img, size_trigger=trigger_size)
                elif attack_type == 'freq':
                    magnitude = 500
                    channel_list = [1, 2]
                    size = window_size = 32
                    trigger_position = [15, 31]
                    poison_frequency_agent = PoisonFre(
                        size, channel_list, window_size, trigger_position,
                        lindct=False, rgb2yuv=True, magnitude=magnitude
                    )
                    poisoned_np = freq_img(img, poison_frequency_agent)
                elif attack_type == 'adapblend':
                    img_np = np.asarray(img)
                    poisoned_np = blend_img(img_np, test_f=False)
                else:
                    raise ValueError(f"Unknown attack_type: {attack_type}")

                poisoned_img = Image.fromarray(poisoned_np)

                img_bytes = encode_pil_to_png_bytes(poisoned_img)
                size_bytes = len(img_bytes)

                poison_bin_f.write(img_bytes)
                image_name = f"trainPoison_{i:05d}.png"

                # CLEAN-LABEL: keep ORIGINAL label (which == poison_target)
                poison_idx_f.write(f"{image_name},{int(label)},{poison_offset},{size_bytes}\n")
                poison_offset += size_bytes

            else:
                # -------------------- CLEAN --------------------
                if i in cover_indices:
                    if attack_type == 'adapblend':
                        img_np = np.asarray(img)
                        poisoned_np = blend_img(img_np, test_f=False)
                        img = Image.fromarray(poisoned_np)
                    elif attack_type == 'pattern_cover':
                        poisoned_np = pattern_img(img, size_trigger=trigger_size)
                        img = Image.fromarray(poisoned_np)
                    else:
                        raise ValueError(f"Unknown attack_type: {attack_type}")

                img_bytes = encode_pil_to_png_bytes(img)
                size_bytes = len(img_bytes)

                clean_bin_f.write(img_bytes)
                image_name = f"trainClean_{i:05d}.png"

                clean_idx_f.write(f"{image_name},{int(label)},{clean_offset},{size_bytes}\n")
                clean_offset += size_bytes

    print("[TRAIN CLEAN] wrote to:")
    print("  ", clean_bin_path)
    print("  ", clean_idx_path)
    print("[TRAIN POISONED] wrote to:")
    print("  ", poison_bin_path)
    print("  ", poison_idx_path)



# -------------------------
# Export clean test split
# -------------------------
def export_clean_test(dataset, out_folder):
    ensure_folder(out_folder)

    bin_path = os.path.join(out_folder, "testSetClear_labels.bin")
    idx_path = os.path.join(out_folder, "testSetClear_labels_index.txt")

    offset = 0
    with open(bin_path, "wb") as bin_f, open(idx_path, "w") as idx_f:
        for i, (img, label) in enumerate(dataset):
            img_bytes = encode_pil_to_png_bytes(img)
            size = len(img_bytes)

            bin_f.write(img_bytes)
            image_name = f"testClean_{i:05d}.png"
            idx_f.write(f"{image_name},{int(label)},{offset},{size}\n")
            offset += size

    print("[TEST CLEAN] wrote to:")
    print("  ", bin_path)
    print("  ", idx_path)


# -------------------------
# Export poisoned test split (all poisoned)
# -------------------------
def export_poisoned_test(dataset, attack_type, out_folder, trigger_size=6):
    ensure_folder(out_folder)

    bin_path = os.path.join(out_folder, "testSetPoisoned_poisoned_list.bin")
    idx_path = os.path.join(out_folder, "testSetPoisoned_poisoned_list_index.txt")

    offset = 0
    with open(bin_path, "wb") as bin_f, open(idx_path, "w") as idx_f:
        for i, (img, label) in enumerate(dataset):
            if attack_type in ['pattern', 'pattern_cover']:
                poisoned_np = pattern_img(img, size_trigger=trigger_size)
                poisoned_img = Image.fromarray(poisoned_np)
            elif attack_type == 'freq':
                magnitude = 500
                channel_list = [1, 2]
                size = window_size = 32
                trigger_position = [15, 31]
                poison_frequency_agent = PoisonFre(
                    size, channel_list, window_size, trigger_position,
                    lindct=False, rgb2yuv=True, magnitude=magnitude
                )
                poisoned_np = freq_img(img, poison_frequency_agent)
                poisoned_img = Image.fromarray(poisoned_np)
            elif attack_type == 'adapblend':
                img_np = np.asarray(img)
                poisoned_np = blend_img(img_np, test_f=True)
                poisoned_img = Image.fromarray(poisoned_np)
            else:
                raise ValueError('No implementation')
            img_bytes = encode_pil_to_png_bytes(poisoned_img)
            size = len(img_bytes)

            bin_f.write(img_bytes)
            image_name = f"testPoison_{i:05d}.png"

            # keep ORIGINAL label; CustomDataset will handle flipping for split_time='test'
            idx_f.write(f"{image_name},{int(label)},{offset},{size}\n")
            offset += size

    print("[TEST POISONED] wrote to:")
    print("  ", bin_path)
    print("  ", idx_path)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Where torchvision will download/load CIFAR-10
    cifar_root = "./cifar_data"
    poison_ratio = 0.05
    attack_type = 'pattern_cover'
    poison_target = 0
    # This is your data_folder / train_data_path
    output_folder = "./datasets/{}_{}_{}/cifar10".format(attack_type, poison_ratio, poison_target)

    # Load CIFAR-10 with no transforms so we get PIL images
    train_set = CIFAR10(root=cifar_root, train=True, download=True, transform=None)
    test_set = CIFAR10(root=cifar_root, train=False, download=True, transform=None)

    # 1) Train: split into 95% clean, 5% poisoned
    export_train_clean_and_poison(
        dataset=train_set,
        attack_type=attack_type,
        poison_target=poison_target,
        out_folder=output_folder,
        poison_ratio=poison_ratio,
        trigger_size=6,
        seed=42,
    )

    # 2) Test: clean
    export_clean_test(
        dataset=test_set,
        out_folder=output_folder,
    )

    # 3) Test: all poisoned
    export_poisoned_test(
        dataset=test_set,
        attack_type=attack_type,
        out_folder=output_folder,
        trigger_size=6,
    )

    print("All exports done.")