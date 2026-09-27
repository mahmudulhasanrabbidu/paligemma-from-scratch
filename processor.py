from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from numpy.typing import DTypeLike
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt: str, bos_token: str, image_seq_len: int, image_token: str) -> str:
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(image: np.ndarray, scale: float, dtype: DTypeLike = np.float32) -> np.ndarray:
    return (image * scale).astype(dtype)


def resize(image: Image.Image, size: Tuple[int, int], resample: Image.Resampling = Image.Resampling.BICUBIC) -> Image.Image:
    height, width = size
    return image.resize((width, height), resample=resample)


def normalize(image: np.ndarray, mean: Union[float, Iterable[float]], std: Union[float, Iterable[float]]) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    return (image - mean) / std


def process_images(
    images: List[Image.Image],
    size: Tuple[int, int] = (224, 224),
    resample: Image.Resampling = Image.Resampling.BICUBIC,
    rescale_factor: float = 1.0 / 255.0,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    if image_mean is None:
        image_mean = IMAGENET_STANDARD_MEAN
    if image_std is None:
        image_std = IMAGENET_STANDARD_STD

    processed = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        # img: [H_orig, W_orig, 3] -> [H, W, 3]
        resized = resize(img, size=size, resample=resample)
        arr = np.array(resized, dtype=np.float32)
        arr = rescale(arr, scale=rescale_factor)
        arr = normalize(arr, mean=image_mean, std=image_std)
        # [H, W, C] -> [C, H, W]
        arr = arr.transpose(2, 0, 1)
        processed.append(arr)
    return processed


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int = 256, image_size: int = 224):
        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        self.tokenizer = tokenizer

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        self.tokenizer.add_special_tokens(tokens_to_add)

        loc_tokens = [f"<loc{i:04d}>" for i in range(1024)]
        seg_tokens = [f"<seg{i:03d}>" for i in range(128)]
        self.tokenizer.add_tokens(loc_tokens + seg_tokens)

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if len(images) != len(text):
            raise ValueError(f"Received {len(images)} images for {len(text)} text prompts.")

        # pixel_arrays: List of [C, H, W]
        pixel_arrays = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1.0 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        # pixel_values: [B, C, H, W]
        pixel_values = torch.tensor(np.stack(pixel_arrays, axis=0), dtype=torch.float32)

        bos = self.tokenizer.bos_token if self.tokenizer.bos_token else "<bos>"
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=bos,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # input_ids: [B, seq_len], attention_mask: [B, seq_len]
        encoded = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
