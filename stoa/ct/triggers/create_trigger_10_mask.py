
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

# 1) load trigger_10.png and make it [3, 32, 32]
trigger = Image.open("trigger_10.png").convert("RGB").resize((32, 32))
trigger_t = to_tensor(trigger)                  # shape: [3, 32, 32], float32 in [0,1]

# 2) create a blank image [3, 224, 224]
img = torch.zeros((3, 224, 224), dtype=trigger_t.dtype)

# 3) attach the trigger at the bottom-right corner
img[:, -32:, -32:] = trigger_t

# (optional) save to check the result
to_pil_image(img).save("base_with_trigger.png")