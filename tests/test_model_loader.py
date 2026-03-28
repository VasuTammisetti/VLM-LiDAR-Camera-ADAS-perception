"""Tests for model loader — no GPU required."""
import pytest


def test_invalid_model_name():
    """Should raise ValueError for unknown model names."""
    # We can't actually load models in CI (no GPU), but we can test validation
    from src.model_loader import load_model
    with pytest.raises(ValueError, match="Unknown model"):
        load_model("nonexistent-model-xyz")


def test_config_has_models():
    """Model configs should be defined."""
    from src.config import MODEL_CONFIGS
    assert "paligemma-3b" in MODEL_CONFIGS
    assert "llava-1.5-7b" in MODEL_CONFIGS
    for name, cfg in MODEL_CONFIGS.items():
        assert "model_id" in cfg
        assert "min_vram_gb" in cfg