"""Tests for scene analyzer — no GPU required."""
from src.scene_analyzer import PROMPTS


class TestPrompts:
    def test_all_keys_exist(self):
        expected = ["full_analysis", "hazard_only", "depth_aware", "object_count"]
        for key in expected:
            assert key in PROMPTS, f"Missing prompt: {key}"

    def test_prompts_not_empty(self):
        for key, prompt in PROMPTS.items():
            assert len(prompt) > 50, f"Prompt '{key}' too short"

    def test_full_analysis_has_sections(self):
        prompt = PROMPTS["full_analysis"]
        assert "Scene Context" in prompt
        assert "Object Detection" in prompt
        assert "Hazard" in prompt
        assert "Driving Recommendation" in prompt

    def test_depth_aware_has_color_legend(self):
        prompt = PROMPTS["depth_aware"]
        assert "Blue" in prompt or "blue" in prompt
        assert "red" in prompt or "Red" in prompt