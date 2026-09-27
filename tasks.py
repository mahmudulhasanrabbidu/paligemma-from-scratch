import os
import re
from typing import Dict, List, Optional, Tuple, Any
import torch
from PIL import Image, ImageDraw, ImageFont

from gemma_new.model import PaliGemmaForConditionalGeneration, KVCache
from gemma_new.processor import PaliGemmaProcessor

# Regular expression to match PaliGemma location tokens: <loc0000> to <loc1023>
LOC_REGEX = re.compile(r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([^<;]+)?")
SEG_REGEX = re.compile(r"<seg(\d{3})>")


def parse_detection_output(raw_text: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    detections = []
    for match in LOC_REGEX.finditer(raw_text):
        y1, x1, y2, x2 = [int(val) for val in match.groups()[:4]]
        label = match.group(5).strip() if match.group(5) else "object"

        # Convert 0-1024 normalized coordinate space to actual pixel coordinates
        ymin = int((y1 / 1024.0) * image_height)
        xmin = int((x1 / 1024.0) * image_width)
        ymax = int((y2 / 1024.0) * image_height)
        xmax = int((x2 / 1024.0) * image_width)

        # Ensure valid box dimensions
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)

        detections.append({
            "box": (xmin, ymin, xmax, ymax),
            "label": label,
            "norm_coords": (y1, x1, y2, x2),
        })

    return detections


def draw_bounding_boxes(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    palette = [
        "#FF3838", "#2F80ED", "#27AE60", "#F2994A",
        "#9B51E0", "#E91E63", "#00BCD4", "#E67E22",
    ]

    for idx, det in enumerate(detections):
        xmin, ymin, xmax, ymax = det["box"]
        label = det["label"]
        color = palette[idx % len(palette)]

        # Draw box outline
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        # Label background pill
        text = f" {label} "
        draw.rectangle([xmin, max(0, ymin - 18), xmin + len(text) * 8, ymin], fill=color)
        draw.text((xmin + 2, max(0, ymin - 16)), text, fill="white")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        annotated.save(output_path)
        print(f"Annotated image saved to: {output_path}")

    return annotated


def parse_segmentation_output(raw_text: str) -> List[int]:
    seg_ids = [int(m.group(1)) for m in SEG_REGEX.finditer(raw_text)]
    return seg_ids


class PaliGemmaPipeline:
    """
    High-level inference engine for PaliGemma supporting all 6 multimodal tasks:
    1. Image Captioning
    2. Visual Question Answering (VQA)
    3. Document Understanding & OCR
    4. Object Detection & Grounding
    5. Instance & Semantic Segmentation
    6. Visual Reasoning & Spatial Analysis
    """
    def __init__(
        self,
        model: PaliGemmaForConditionalGeneration,
        processor: PaliGemmaProcessor,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device
        self.stop_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 1

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 0.0,
        top_p: float = 0.9,
        do_sample: bool = False,
        verbose: bool = True,
    ) -> str:
        """
        Runs multimodal inference with KV-caching.
        """
        if verbose:
            print(f"\n[Prompt]: '{prompt}'")

        inputs = self.processor(text=[prompt], images=[image])
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.model.config.text_config.dtype)

        kv_cache = KVCache()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )

        logits = outputs["logits"][:, -1, :]
        if do_sample and temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            # Nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            sorted_probs[sorted_indices_to_remove] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_indices.gather(-1, torch.multinomial(sorted_probs, 1))
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated_tokens = [next_token.item()]
        if verbose:
            first_decoded = self.tokenizer.decode([next_token.item()])
            print(f"[Generating]: {first_decoded}", end="", flush=True)

        curr_input_ids = next_token
        curr_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)], dim=-1
        )

        for step in range(max_new_tokens - 1):
            if next_token.item() == self.stop_token_id:
                break

            with torch.no_grad():
                step_outputs = self.model(
                    input_ids=curr_input_ids,
                    pixel_values=None,
                    attention_mask=curr_attention_mask,
                    kv_cache=kv_cache,
                )

            step_logits = step_outputs["logits"][:, -1, :]
            if do_sample and temperature > 0.0:
                probs = torch.softmax(step_logits / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                sorted_probs[sorted_indices_to_remove] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_indices.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = torch.argmax(step_logits, dim=-1, keepdim=True)

            generated_tokens.append(next_token.item())
            if verbose:
                piece = self.tokenizer.decode([next_token.item()])
                print(piece, end="", flush=True)

            curr_input_ids = next_token
            curr_attention_mask = torch.cat(
                [curr_attention_mask, torch.ones((1, 1), dtype=curr_attention_mask.dtype, device=self.device)], dim=-1
            )

        if verbose:
            print()

        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return decoded

    def caption(self, image: Image.Image, lang: str = "en", detailed: bool = False, max_tokens: int = 35) -> str:
        """Generates a caption for the image."""
        prefix = "describe" if detailed else "caption"
        prompt = f"{prefix} {lang}"
        return self.generate(image, prompt=prompt, max_new_tokens=max_tokens)

    def vqa(self, image: Image.Image, question: str, max_tokens: int = 30) -> str:
        """Answers a question about the image."""
        # PaliGemma was fine-tuned on questions directly or with 'answer en'
        prompt = question if question.lower().startswith("answer") else f"answer en {question}"
        return self.generate(image, prompt=prompt, max_new_tokens=max_tokens)

    def ocr(self, image: Image.Image, max_tokens: int = 60) -> str:
        """Extracts text, signs, labels, or document contents from the image."""
        prompt = "ocr"
        return self.generate(image, prompt=prompt, max_new_tokens=max_tokens)

    def detect(
        self,
        image: Image.Image,
        target: Optional[str] = None,
        max_tokens: int = 50,
        draw_boxes: bool = True,
        save_path: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Optional[Image.Image]]:
        """
        Detects objects in the image, extracts normalized bounding box coordinates,
        and optionally draws boxes onto the image.
        """
        prompt = f"detect {target}" if target else "detect"
        raw_output = self.generate(image, prompt=prompt, max_new_tokens=max_tokens)
        detections = parse_detection_output(raw_output, image.width, image.height)

        annotated_img = None
        if draw_boxes:
            annotated_img = draw_bounding_boxes(image, detections, output_path=save_path)

        return raw_output, detections, annotated_img

    def segment(
        self,
        image: Image.Image,
        target: Optional[str] = None,
        max_tokens: int = 50,
    ) -> Tuple[str, List[int]]:
        """
        Generates segmentation codebook tokens for the requested object.
        """
        prompt = f"segment {target}" if target else "segment"
        raw_output = self.generate(image, prompt=prompt, max_new_tokens=max_tokens)
        seg_ids = parse_segmentation_output(raw_output)
        return raw_output, seg_ids

    def visual_reasoning(self, image: Image.Image, question: str, max_tokens: int = 35) -> str:
        """
        Performs spatial reasoning, counting, or attribute comparison on the image.
        """
        return self.vqa(image, question=question, max_tokens=max_tokens)
