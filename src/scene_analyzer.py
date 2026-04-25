"""
ADAS-specific prompt templates and VLM inference.
"""
import torch

PROMPTS = {
    "full_analysis": (
        "You are an ADAS perception system analyzing a driving scene.\n"
        "Provide a structured analysis:\n"
        "1. Scene Context: Road type, weather, lighting, lane configuration\n"
        "2. Object Detection: List all road users with approximate distance and position\n"
        "3. Hazard Assessment: Identify potential risks ranked by severity (critical/high/medium/low)\n"
        "4. Driving Recommendation: Suggest appropriate driving action\n"
        "Be precise and safety-focused. Use approximate distances in meters."
    ),
    "hazard_only": (
        "As an autonomous driving safety system, identify ALL potential hazards "
        "in this driving scene. For each hazard specify:\n"
        "- Type (pedestrian, vehicle, cyclist, obstacle, road condition)\n"
        "- Location (left/center/right, approximate distance in meters)\n"
        "- Risk level (critical/high/medium/low)\n"
        "- Recommended action"
    ),
    "depth_aware": (
        "This driving scene has LiDAR depth overlay. "
        "The colored dots represent LiDAR points:\n"
        "- Blue/purple = close (0-10m)\n"
        "- Green/yellow = mid-range (10-25m)\n"
        "- Orange/red = far (25-50m)\n\n"
        "Analyze:\n"
        "1. Scene description with road layout\n"
        "2. All detected objects with ESTIMATED DISTANCES using the depth colors\n"
        "3. Hazard assessment with distance-based priority\n"
        "4. Driving recommendation"
    ),
    "object_count": (
        "Count and list every object in this driving scene. "
        "For each object provide: type, position (left/center/right), "
        "estimated distance, and whether it is moving or stationary."
    ),
}


def analyze_scene(image, model, processor, prompt_type="full_analysis", max_tokens=512):
    """
    Run VLM inference on a driving scene image.

    Args:
        image: PIL Image
        model: loaded VLM model
        processor: model processor/tokenizer
        prompt_type: key from PROMPTS dict
        max_tokens: max generation length

    Returns:
        str: model's analysis text
    """
    prompt = PROMPTS[prompt_type]
    model_name = model.config._name_or_path.lower()

    # Build input based on model type
    if "paligemma" in model_name:
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(
            model.device
        )
    else:
        # LLaVA-style chat template
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt").to(
            model.device
        )

    # Inference
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.2, do_sample=True
        )

    # Decode and clean
    decoded = processor.decode(output[0], skip_special_tokens=True)

    # Strip the input prompt from output
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()
    elif prompt[:50] in decoded:
        decoded = decoded.split(prompt)[-1].strip()

    return decoded


def batch_analyze(image_paths, model, processor, prompt_type="full_analysis"):
    """
    Analyze multiple scenes sequentially.

    Args:
        image_paths: list of image file paths
        model, processor: loaded model
        prompt_type: prompt key

    Returns:
        dict: {filename: analysis_text}
    """
    from PIL import Image

    results = {}
    for i, path in enumerate(image_paths):
        print(f"  Analyzing [{i+1}/{len(image_paths)}]: {path}")
        img = Image.open(path).convert("RGB")
        result = analyze_scene(img, model, processor, prompt_type)
        results[path] = result
        torch.cuda.empty_cache()
    return results
