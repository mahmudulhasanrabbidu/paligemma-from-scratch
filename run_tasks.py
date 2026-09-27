import os
import sys
import argparse
import torch
from PIL import Image
from transformers import AutoTokenizer

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemma_new.pretrained_weight_loader import PretrainedPaliGemmaModel
from gemma_new.processor import PaliGemmaProcessor
from gemma_new.tasks import PaliGemmaPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run PaliGemma Multimodal Tasks (Captioning, VQA, OCR, Detection, Segmentation, Spatial Reasoning)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="paligemma-weights/paligemma-3b-mix-224",
        help="Path to downloaded PaliGemma model weights",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="test_images/pic1.jpeg",
        help="Path to input image",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["all", "caption", "vqa", "ocr", "detect", "segment", "reason"],
        default="all",
        help="Multimodal task to perform",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Where is the photographer resting?",
        help="Question for VQA or visual reasoning",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target class for detection or segmentation (e.g. 'person', 'dog', 'camera')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save annotated images and results",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model directory '{args.model_path}' not found.")
        sys.exit(1)

    if not os.path.exists(args.image_path):
        print(f"Error: Image '{args.image_path}' not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PaliGemma Multimodal Task Suite")
    print(f"Model: {args.model_path}")
    print(f"Image: {args.image_path}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Load Pretrained PaliGemma Model
    model, _ = PretrainedPaliGemmaModel.from_pretrained(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.bfloat16,
    )
    model.eval()

    # Setup Tokenizer and Processor
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = PaliGemmaProcessor(
        tokenizer=tokenizer,
        num_image_tokens=model.config.vision_config.num_image_tokens,
        image_size=model.config.vision_config.image_size,
    )

    # Create Pipeline
    pipeline = PaliGemmaPipeline(model=model, processor=processor, device=args.device)
    raw_image = Image.open(args.image_path)

    # Execute Selected Task(s)
    tasks_to_run = (
        ["caption", "vqa", "ocr", "detect", "segment", "reason"]
        if args.task == "all"
        else [args.task]
    )

    print("\nExecuting Tasks...")
    for t in tasks_to_run:
        print(f"\n{'='*30} TASK: {t.upper()} {'='*30}")

        if t == "caption":
            # Task A: Image Captioning
            print("Generating English Caption...")
            caption_en = pipeline.caption(raw_image, lang="en", max_tokens=30)
            print(f"Result: {caption_en}")

        elif t == "vqa":
            # Task B: Visual Question Answering
            q = args.question
            print(f"Question: {q}")
            answer = pipeline.vqa(raw_image, question=q, max_tokens=25)
            print(f"Answer: {answer}")

        elif t == "ocr":
            # Task C: Document Understanding & OCR
            print("Reading text from image (OCR)...")
            ocr_text = pipeline.ocr(raw_image, max_tokens=40)
            print(f"Detected Text: {ocr_text if ocr_text else '(No text detected in image)'}")

        elif t == "detect":
            # Task D: Object Detection
            detect_label = args.target if args.target else ""
            save_img_path = os.path.join(args.output_dir, "annotated_detection.jpg")
            print(f"Detecting objects ({detect_label if detect_label else 'all'})...")
            raw_out, detections, annotated_img = pipeline.detect(
                raw_image,
                target=detect_label,
                max_tokens=40,
                draw_boxes=True,
                save_path=save_img_path,
            )
            print(f"Raw Output: {raw_out}")
            print(f"Detected {len(detections)} object(s):")
            for d in detections:
                print(f"  - Label: '{d['label']}' | Bounding Box: {d['box']}")

        elif t == "segment":
            # Task E: Segmentation
            target_obj = args.target if args.target else "person"
            print(f"Segmenting object '{target_obj}'...")
            raw_out, seg_tokens = pipeline.segment(raw_image, target=target_obj, max_tokens=40)
            print(f"Raw Output: {raw_out}")
            print(f"Extracted {len(seg_tokens)} segmentation codebook token(s): {seg_tokens[:10]}...")

        elif t == "reason":
            # Task F: Visual Reasoning & Spatial Analysis
            reason_q = "What is the person doing in the photo?"
            print(f"Spatial/Visual Reasoning Question: '{reason_q}'")
            reason_res = pipeline.visual_reasoning(raw_image, question=reason_q, max_tokens=30)
            print(f"Reasoning Result: {reason_res}")

    print("\n" + "=" * 70)
    print("ALL REQUESTED TASKS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
