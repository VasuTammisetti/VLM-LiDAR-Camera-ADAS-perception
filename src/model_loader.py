"""
Model loading with 4-bit quantization for low-VRAM GPUs.
"""
import torch

def get_device_info():
    """Print GPU info and return available VRAM."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {name} | VRAM: {vram:.1f} GB")
        return vram
    else:
        print("No GPU detected. Running on CPU (very slow).")
        return 0

def load_model(model_name="llava-1.5-7b"):
    """
    Load VLM with 4-bit quantization.

    Args:
        model_name: 'llava-1.5-7b' (recommended) or 'paligemma-3b'

    Returns:
        model, processor tuple
    """
    from transformers import BitsAndBytesConfig

    vram = get_device_info()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    if model_name == "llava-1.5-7b":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        print(f"Loading {model_id} (4-bit)...")

        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

    elif model_name == "paligemma-3b":
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        model_id = "google/paligemma-3b-mix-448"
        print(f"Loading {model_id} (4-bit)...")

        processor = AutoProcessor.from_pretrained(model_id)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'llava-1.5-7b' or 'paligemma-3b'")

    used = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded! VRAM used: {used:.1f} GB")
    return model, processor
