import os
import sys

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
from transformers import AutoTokenizer

from gemma_new.pretrained_weight_loader import PretrainedPaliGemmaModel
from gemma_new.processor import PaliGemmaProcessor
from gemma_new.model import KVCache


def test_paligemma_pipeline():
    print("=" * 70)
    print("TESTING FULL PALIGEMMA MULTIMODAL WEIGHT LOADING & INFERENCE")
    print("=" * 70)

    model_dir = "paligemma-weights/paligemma-3b-mix-224"
    image_path = "test_images/pic1.jpeg"
    prompt = "caption en"

    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Error: Test image '{image_path}' not found.")
        sys.exit(1)

    print("\n--- 1. Loading Pretrained PaliGemma Model ---")
    model, assigned_keys = PretrainedPaliGemmaModel.from_pretrained(
        model_path=model_dir,
        device="cpu",
        dtype=torch.bfloat16,
    )
    model.eval()

    print("\n--- 2. Setting up Tokenizer and Processor ---")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = PaliGemmaProcessor(
        tokenizer=tokenizer,
        num_image_tokens=model.config.vision_config.num_image_tokens,
        image_size=model.config.vision_config.image_size,
    )

    print(f"\n--- 3. Processing '{image_path}' with prompt: '{prompt}' ---")
    raw_image = Image.open(image_path)
    inputs = processor(text=[prompt], images=[raw_image])

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)

    print(f"  pixel_values shape: {pixel_values.shape} | dtype: {pixel_values.dtype}")
    print(f"  input_ids shape:    {input_ids.shape} | prompt tokens: {input_ids.shape[1]}")
    print(f"  attention_mask:     {attention_mask.shape}")

    print("\n--- 4. Executing Prefill Forward Pass ---")
    kv_cache = KVCache()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

    logits = outputs["logits"]
    print(f"  Output logits shape: {logits.shape}")
    assert logits.shape == (1, input_ids.shape[1], model.config.vocab_size)
    assert not torch.isnan(logits).any(), "NaN found in prefill logits!"
    print("  Logits verification PASSED (valid values, no NaNs).")

    print("\n--- 5. Generating Caption (Greedy Decoding) ---")
    max_new_tokens = 25
    stop_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    generated_tokens = []
    # Extract prediction for the last prompt token
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_tokens.append(next_token.item())

    # Decode loop
    curr_input_ids = next_token
    curr_attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)], dim=-1
    )

    for step in range(max_new_tokens - 1):
        if next_token.item() == stop_token_id:
            break

        with torch.no_grad():
            step_outputs = model(
                input_ids=curr_input_ids,
                pixel_values=None,
                attention_mask=curr_attention_mask,
                kv_cache=kv_cache,
            )

        step_logits = step_outputs["logits"][:, -1, :]
        next_token = torch.argmax(step_logits, dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())

        curr_input_ids = next_token
        curr_attention_mask = torch.cat(
            [curr_attention_mask, torch.ones((1, 1), dtype=curr_attention_mask.dtype)], dim=-1
        )

    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print(f"\n========================================================")
    print(f"PROMPT:             {prompt}")
    print(f"GENERATED CAPTION:  {decoded_text}")
    print(f"========================================================")
    print("\nALL PALIGEMMA TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    test_paligemma_pipeline()
