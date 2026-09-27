import os
import sys
import torch

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemma_new.pretrained_weight_loader import PretrainedGemmaModel

def main():
    model_dir = "paligemma-weights/paligemma-3b-mix-224"
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        sys.exit(1)

    print("=" * 70)
    print("Testing PretrainedGemmaModel weight loading (Reasoning-LLM style)")
    print("=" * 70)

    device = "cpu"
    model, _ = PretrainedGemmaModel.from_pretrained(
        model_path=model_dir,
        device=device,
    )

    model.visualize_custom_parameters()

    print("Running forward pass test with dummy token prompt...")
    model.eval()
    
    # Input tokens: [batch_size=1, seq_len=5]
    sample_input_ids = torch.tensor([[2, 106, 1645, 108, 1]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        outputs = model(input_ids=sample_input_ids)
        logits = outputs["logits"]

    print(f"Forward pass output logits shape: {logits.shape}")
    assert logits.shape == (1, 5, model.config.vocab_size), (
        f"Expected shape (1, 5, {model.config.vocab_size}), got {logits.shape}"
    )
    assert not torch.isnan(logits).any(), "Logits contain NaN values!"

    print("\n[SUCCESS] Pretrained weight loader and forward pass verified successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
