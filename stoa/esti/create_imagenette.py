import os
import io
import random

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import Imagenette   # <-- CHANGED
from torchvision.transforms import ToTensor, ToPILImage
from lib.frequency import PoisonFre
from lib.rawDataProcessing import freq_img, blend_img
from torchvision.transforms import transforms, ToTensor, ToPILImage, Compose, Resize

# -------------------------
# Your trigger function (unchanged)
# -------------------------
def pattern_img(image):
    """
    image tensor with pixel value from [0, 1]
    :param image: image tensor with shape [channel, width, height]
    :param max: the maximums for three different channels
    :param min: the minimums for three different channels
    :return: image tensor with trigger
    """
    patch_size = 90  # the trigger's width and height (dividable by 3)
    trigger_path = 'trigger_10.png'
    trigger = Compose([
        Resize((patch_size, patch_size)),
        ToTensor(),
    ])(Image.open(trigger_path).convert('RGB'))

    image_size = image.size[0]  # assuming square images
    # Choose random location to inject the trigger
    start_x = image_size - patch_size - 20 # random.randint(0, image_size - patch_size - 1)
    start_y = image_size - patch_size - 20 # random.randint(0, image_size - patch_size - 1)
    To_Tensor = ToTensor()
    img_tensor = To_Tensor(image)
    # Inject trigger (self.trigger must be a torch.Tensor of shape (C, patch_size, patch_size))
    img_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = trigger

    # Convert back to PIL image
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_tensor)

    img_np = np.array(img_pil)

    return img_np
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
# Export train split: clean + poisoned
# -------------------------


def export_train_clean_and_poison(
    dataset,
    attack_type,
    out_folder,
    poison_target,
    poison_ratio=0.05,
    seed=42,
):
    """
    dataset: Imagenette(split='train', transform=None)
    attack_type: 'pattern', 'freq', 'adapblend', or 'pattern_cover'
    out_folder: where to write .bin and _index.txt
    poison_target: int, target class for clean-label attack
    poison_ratio: fraction of *entire* train set to be poisoned
    """

    ensure_folder(out_folder)

    clean_bin_path = os.path.join(out_folder, "trainSet_clean_list.bin")
    clean_idx_path = os.path.join(out_folder, "trainSet_clean_list_index.txt")

    poison_bin_path = os.path.join(out_folder, "trainSet_poisoned_list.bin")
    poison_idx_path = os.path.join(out_folder, "trainSet_poisoned_list_index.txt")

    n = len(dataset)
    n_poison = int(poison_ratio * n)

    # ------------------------------------------------------------------
    # 1) Select poison indices depending on attack_type
    # ------------------------------------------------------------------
    if attack_type == 'freq':
        # clean-label: only poison samples of the target class
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
    print(f"Target class: {poison_target}, total target samples (poison_indices): {len(poison_indices)}")

    clean_offset = 0
    poison_offset = 0

    with open(clean_bin_path, "wb") as clean_bin_f, \
         open(clean_idx_path, "w") as clean_idx_f, \
         open(poison_bin_path, "wb") as poison_bin_f, \
         open(poison_idx_path, "w") as poison_idx_f:

        for i, (img, label) in enumerate(dataset):
            # img resize to 224 by 224
            img = img.resize((224, 224))
            if i in poison_indices:
                # -------------------- POISONED --------------------
                if attack_type in ['pattern', 'pattern_cover']:
                    poisoned_np = pattern_img(img)
                    poisoned_img = Image.fromarray(poisoned_np)
                elif attack_type == 'freq':
                    # NOTE: For Imagenette, images are larger than 32x32.
                    # You may want to adapt these parameters to your PoisonFre implementation.
                    channel_list = [1, 2]
                    size = window_size = 224
                    trigger_position = [15, 31]
                    magnitude = 5000
                    poison_frequency_agent = PoisonFre(
                        size, channel_list, window_size, trigger_position,
                        lindct=False, rgb2yuv=True, magnitude=magnitude
                    )
                    poisoned_np = freq_img(img, poison_frequency_agent)
                    poisoned_img = Image.fromarray(poisoned_np)
                elif attack_type == 'adapblend':
                    img_np = np.asarray(img)
                    poisoned_np = blend_img(img_np, test_f=False, alpha=0.3)
                    poisoned_img = Image.fromarray(poisoned_np)
                else:
                    raise ValueError(f"Unknown attack_type: {attack_type}")

                img_bytes = encode_pil_to_png_bytes(poisoned_img)
                size_bytes = len(img_bytes)

                poison_bin_f.write(img_bytes)
                image_name = f"trainPoison_{i:05d}.png"

                # CLEAN-LABEL: keep ORIGINAL label
                poison_idx_f.write(f"{image_name},{int(label)},{poison_offset},{size_bytes}\n")
                poison_offset += size_bytes

            else:
                # -------------------- CLEAN --------------------
                if i in cover_indices:
                    if attack_type == 'adapblend':
                        img_np = np.asarray(img)
                        poisoned_np = blend_img(img_np, test_f=False, alpha=0.3)
                        img = Image.fromarray(poisoned_np)
                    elif attack_type == 'pattern_cover':
                        poisoned_np = pattern_img(img)
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
            img = img.resize((224, 224))
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
def export_poisoned_test(dataset, attack_type, out_folder):
    ensure_folder(out_folder)

    bin_path = os.path.join(out_folder, "testSetPoisoned_poisoned_list.bin")
    idx_path = os.path.join(out_folder, "testSetPoisoned_poisoned_list_index.txt")

    offset = 0
    with open(bin_path, "wb") as bin_f, open(idx_path, "w") as idx_f:
        for i, (img, label) in enumerate(dataset):
            img = img.resize((224, 224))
            if attack_type in ['pattern', 'pattern_cover']:
                poisoned_np = pattern_img(img)
                poisoned_img = Image.fromarray(poisoned_np)
            elif attack_type == 'freq':
                # You may want to adapt these parameters to your PoisonFre implementation.
                channel_list = [1, 2]
                size = window_size = 224
                trigger_position = [15, 31]
                magnitude = 5000

                poison_frequency_agent = PoisonFre(
                    size, channel_list, window_size, trigger_position,
                    lindct=False, rgb2yuv=True, magnitude=magnitude
                )
                poisoned_np = freq_img(img, poison_frequency_agent)
                poisoned_img = Image.fromarray(poisoned_np)
            elif attack_type == 'adapblend':
                img_np = np.asarray(img)
                poisoned_np = blend_img(img_np, test_f=True, alpha=0.3)
                poisoned_img = Image.fromarray(poisoned_np)
            else:
                raise ValueError('No implementation')

            img_bytes = encode_pil_to_png_bytes(poisoned_img)
            size = len(img_bytes)

            bin_f.write(img_bytes)
            image_name = f"testPoison_{i:05d}.png"

            # keep ORIGINAL label
            idx_f.write(f"{image_name},{int(label)},{offset},{size}\n")
            offset += size

    print("[TEST POISONED] wrote to:")
    print("  ", bin_path)
    print("  ", idx_path)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Where torchvision will download/load Imagenette
    imagenette_root = "./imagenette_data"           # <-- CHANGED
    poison_ratio = 0.003
    attack_type = 'freq'                  # works fine with Imagenette
    poison_target = 0                              # Imagenette has 10 classes (0..9)

    # This is your data_folder / train_data_path
    output_folder = "./datasets/{}_{}_{}/imagenette".format(  # <-- CHANGED name
        attack_type, poison_ratio, poison_target
    )

    # Load Imagenette with no transforms so we get PIL images
    # size can be "full", "320px" or "160px"
    train_set = Imagenette(
        root=imagenette_root,
        split="train",
        size="160px",     # smaller images; change to "full"/"320px" if you want
        download=False,
        transform=None,
    )
    test_set = Imagenette(
        root=imagenette_root,
        split="val",
        size="160px",
        download=False,
        transform=None,
    )

    # 1) Train: split into clean + poisoned
    export_train_clean_and_poison(
        dataset=train_set,
        attack_type=attack_type,
        poison_target=poison_target,
        out_folder=output_folder,
        poison_ratio=poison_ratio,
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
    )

    print("All exports done.")
