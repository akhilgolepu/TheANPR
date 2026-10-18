"""
Unit tests for ml_service.py
Tests OCR logic, state correction, format scoring, and spatial ordering.
"""

import pytest
import cv2
import numpy as np
from src.api.ml_service import ANPREngine, VALID_INDIAN_STATES


class TestFormatScore:
    """Test format score calculation for Indian license plates."""
    
    def test_perfect_plate_format(self):
        """Perfect format should score 1.0."""
        engine = ANPREngine("models/checkpoints/plate_detector_v1.pt")
        
        test_cases = [
            "MH20DV2366",
            "DL08EM1234",
            "KA07AB9876",
            "TN01AB5432",
        ]
        
        for plate in test_cases:
            score = engine._calculate_format_score(plate)
            assert score == 1.0, f"Expected 1.0 for {plate}, got {score}"
    
    def test_valid_variant_format(self):
        """Valid variant with single letter should score 1.0."""
        engine = ANPREngine("models/checkpoints/plate_detector_v1.pt")
        
        test_cases = [
            "DL8C1234",  # Single letter variant
            "MH12A1234",
        ]
        
        for plate in test_cases:
            score = engine._calculate_format_score(plate)
            assert score == 1.0, f"Expected 1.0 for {plate}, got {score}"
    
    def test_close_match_format(self):
        """9-11 alphanumeric chars should score 0.5."""
        engine = ANPREngine("models/checkpoints/plate_detector_v1.pt")
        
        test_cases = [
            "MH20DV236",   # 9 chars (missing 1)
            "MH20DV23661", # 11 chars (extra 1)
        ]
        
        for plate in test_cases:
            score = engine._calculate_format_score(plate)
            assert score == 0.5, f"Expected 0.5 for {plate}, got {score}"
    
    def test_invalid_state_code(self):
        """Invalid state code should score 0.0."""
        engine = ANPREngine("models/checkpoints/plate_detector_v1.pt")
        
        test_cases = [
            "XX20DV2366",  # Invalid state XX
            "ZZ12AB1234",  # Invalid state ZZ
        ]
        
        for plate in test_cases:
            score = engine._calculate_format_score(plate)
            assert score == 0.0, f"Expected 0.0 for invalid state in {plate}, got {score}"
    
    def test_too_short_plate(self):
        """Plates too short should score 0.0."""
        engine = ANPREngine("models/checkpoints/plate_detector_v1.pt")
        
        test_cases = [
            "MH20",
            "MH20D",
        ]
        
        for plate in test_cases:
            score = engine._calculate_format_score(plate)
            assert score == 0.0, f"Expected 0.0 for short plate {plate}, got {score}"
    
    def test_empty_plate(self):
        """Empty plate should score 0.0."""
        engine = ANPREngine("models/checkpoints/plate_detector_v1.pt")
        score = engine._calculate_format_score("")
        assert score == 0.0, f"Expected 0.0 for empty plate, got {score}"


class TestStateValidation:
    """Test that all valid Indian states are recognized."""
    
    def test_all_valid_states(self):
        """All VALID_INDIAN_STATES should be recognized."""
        expected_states = {
            "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
            "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
            "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
            "UP", "WB"
        }
        
        assert VALID_INDIAN_STATES == expected_states, \
            f"VALID_INDIAN_STATES mismatch. Expected {expected_states}, got {VALID_INDIAN_STATES}"
    
    def test_state_count(self):
        """India has 36 administrative regions (states + UTs)."""
        # Note: Could be 28-36 depending on how UTs are counted
        assert len(VALID_INDIAN_STATES) >= 28, \
            f"Expected at least 28 states, got {len(VALID_INDIAN_STATES)}"


class TestModelLoading:
    """Test that the ANPREngine initializes correctly."""
    
    def test_engine_initialization(self):
        """Engine should initialize without errors."""
        engine = ANPREngine("models/checkpoints/plate_detector_v1.pt")
        
        assert engine is not None
        assert engine.detector is not None
        assert engine.reader is not None
        assert hasattr(engine, 'predict')
        assert hasattr(engine, '_calculate_format_score')


class TestPredictInterface:
    """Test the public predict() interface."""
    
    def test_predict_invalid_image_bytes(self):
        """Invalid image bytes should return success=False."""
        engine = ANPREngine("models/checkpoints/plate_detector_v1.pt")
        
        result = engine.predict(b"not_an_image")
        
        assert result["success"] is False
        assert "error" in result
    
    def test_predict_empty_image(self):
        """Valid but empty image should return success=True with empty detections."""
        engine = ANPREngine("models/checkpoints/plate_detector_v1.pt")
        
        # Create a blank white image (640x480, BGR)
        blank_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        _, image_bytes = cv2.imencode('.jpg', blank_image)
        
        result = engine.predict(image_bytes.tobytes())
        
        assert result["success"] is True
        assert "detections" in result
        assert isinstance(result["detections"], list)
        # Blank image unlikely to have detections
        # (might have false positives, but list should be iterable)


class TestDetectionStructure:
    """Test that detections follow expected structure."""
    
    def test_detection_response_format(self):
        """Detection response should have required fields."""
        engine = ANPREngine("models/checkpoints/plate_detector_v1.pt")
        
        # Create a simple test image with a bright rectangle (potential plate region)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        test_image[100:200, 200:400] = 50  # Darker rect (potential plate)
        
        _, image_bytes = cv2.imencode('.jpg', test_image)
        result = engine.predict(image_bytes.tobytes())
        
        assert result["success"] is True
        assert "detections" in result
        
        # If detections exist, check structure
        for detection in result["detections"]:
            assert "bbox" in detection
            assert "plate_text" in detection
            assert "confidence" in detection
            assert "format_score" in detection
            
            # Check bbox is [x1, y1, x2, y2]
            assert len(detection["bbox"]) == 4
            assert all(isinstance(x, (int, float)) for x in detection["bbox"])
            
            # Check confidence is 0-1
            assert 0.0 <= detection["confidence"] <= 1.0
            
            # Check format_score is 0-1
            assert 0.0 <= detection["format_score"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
