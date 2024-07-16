from transparent_background import Remover
from PIL import Image
import torch
import numpy as np

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def create_forepil_from_mask(input_image_pil, mask_image_pil):
    input_image = np.array(input_image_pil)
    mask = np.array(mask_image_pil.convert('L'))  # Ensure mask is grayscale

    # Resize mask to match input image dimensions
    mask_resized = np.array(mask_image_pil.resize((input_image.shape[1], input_image.shape[0]), Image.LANCZOS))

    # Normalize mask to range [0, 1]
    mask_normalized = mask_resized / 255.0

    # Initialize alpha channel separately
    alpha_channel = (255 * mask_normalized).astype(np.uint8)

    # Create foreground image with alpha channel
    foreground = np.concatenate((input_image, alpha_channel[:, :, np.newaxis]), axis=2)

    # Convert result back to PIL Image
    foreground_pil = Image.fromarray(foreground, 'RGBA')

    return foreground_pil

class TransparentBackgroundRembg:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("PATH"),
                "image": ("IMAGE"),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"


    def remove_background(self, image, model_path):
        remover = Remover(mode='base-nightly', ckpt=model_path, device='cuda:0')

        image_pil = tensor2pil(image)
        mask = remover.process(image_pil, type='map')
        mask_tensor = pil2tensor(mask)
        fore_pil = create_forepil_from_mask(image, mask)
        fore_tensor = pil2tensor(fore_pil)

        return (fore_tensor, mask_tensor)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Transparentbackground RemBg": TransparentBackgroundRembg
}
