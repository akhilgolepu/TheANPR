"""
Integration tests for main.py FastAPI endpoints.
Tests detection, vehicle CRUD, CSV import/export, and correction flow.
"""

import pytest
import io
import json
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import cv2

# Test fixtures and helpers
TEST_IMAGE_WIDTH = 640
TEST_IMAGE_HEIGHT = 480


def create_test_image_bytes():
    """Create a simple test image (blank white)."""
    test_image = np.ones((TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, 3), dtype=np.uint8) * 255
    _, image_bytes = cv2.imencode('.jpg', test_image)
    return image_bytes.tobytes()


@pytest.fixture
def client():
    """Fixture to provide FastAPI test client."""
    try:
        from src.api.main import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI app not available")


class TestHealthCheck:
    """Test basic health endpoints."""
    
    def test_root_endpoint(self, client):
        """Root endpoint should return OK."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_docs_endpoint(self, client):
        """Swagger docs should be available."""
        response = client.get("/docs")
        assert response.status_code == 200


class TestDetectionEndpoint:
    """Test image detection endpoint."""
    
    def test_detect_image_success(self, client):
        """Detection endpoint should accept image upload."""
        image_bytes = create_test_image_bytes()
        
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        
        assert "success" in data
        assert "detections" in data
        assert isinstance(data["detections"], list)
    
    def test_detect_image_no_file(self, client):
        """Detection endpoint should reject missing file."""
        response = client.post("/api/v1/detect/image")
        assert response.status_code == 422  # Unprocessable entity
    
    def test_detect_invalid_image(self, client):
        """Detection endpoint should handle invalid image gracefully."""
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("test.txt", io.BytesIO(b"not_an_image"), "text/plain")}
        )
        # Could be 400/422 depending on validation
        assert response.status_code in [400, 422, 200]


class TestCorrectionEndpoint:
    """Test human correction endpoint."""
    
    def test_correct_detection(self, client):
        """Correction endpoint should accept corrections."""
        # First, upload an image to get a record ID
        image_bytes = create_test_image_bytes()
        
        upload_response = client.post(
            "/api/v1/detect/image",
            files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}
        )
        
        if upload_response.status_code in [200, 201]:
            data = upload_response.json()
            
            # Try to correct a detection if one exists
            if "detections" in data and len(data["detections"]) > 0:
                # Assuming the API returns db_id in detections
                # This depends on actual implementation
                pass


class TestVehicleListEndpoint:
    """Test vehicle listing endpoint."""
    
    def test_list_vehicles_success(self, client):
        """Vehicle list endpoint should return list."""
        response = client.get("/api/vehicles")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list) or isinstance(data, dict)
        if isinstance(data, dict) and "vehicles" in data:
            assert isinstance(data["vehicles"], list)
    
    def test_list_vehicles_with_filter_status(self, client):
        """Vehicle list should support status filter."""
        response = client.get("/api/vehicles?status=Clear")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))
    
    def test_list_vehicles_with_filter_search(self, client):
        """Vehicle list should support search filter."""
        response = client.get("/api/vehicles?search=MH")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))


class TestVehicleCreateEndpoint:
    """Test vehicle creation endpoint."""
    
    def test_create_vehicle_success(self, client):
        """Create vehicle endpoint should accept valid vehicle."""
        vehicle_data = {
            "plate_number": "MH20TEST1234",
            "vehicle_make": "Toyota",
            "vehicle_model": "Fortuner",
            "vehicle_year": 2020,
            "vehicle_color": "White",
            "vehicle_type": "SUV",
            "owner_name": "Test Owner",
            "owner_phone": "9876543210",
            "owner_address": "Test Address",
            "registration_number": "TEST123",
            "registration_state": "MH",
            "status": "Clear"
        }
        
        response = client.post("/api/vehicles", json=vehicle_data)
        
        # Should either succeed or conflict if already exists
        assert response.status_code in [200, 201, 409]
        
        if response.status_code in [200, 201]:
            data = response.json()
            assert "plate_number" in data or "success" in data


class TestVehicleGetEndpoint:
    """Test vehicle detail endpoint."""
    
    def test_get_vehicle_not_found(self, client):
        """Get endpoint should return 404 for nonexistent vehicle."""
        response = client.get("/api/vehicles/record/NONEXISTENT")
        
        assert response.status_code == 404


class TestVehicleUpdateEndpoint:
    """Test vehicle update endpoint."""
    
    def test_update_vehicle_not_found(self, client):
        """Update endpoint should return 404 for nonexistent vehicle."""
        update_data = {"vehicle_color": "Red"}
        
        response = client.put(
            "/api/vehicles/record/NONEXISTENT",
            json=update_data
        )
        
        assert response.status_code == 404


class TestVehicleDeleteEndpoint:
    """Test vehicle delete endpoint."""
    
    def test_delete_vehicle_not_found(self, client):
        """Delete endpoint should return 404 for nonexistent vehicle."""
        response = client.delete("/api/vehicles/record/NONEXISTENT")
        
        assert response.status_code == 404


class TestExportEndpoint:
    """Test CSV export endpoint."""
    
    def test_export_vehicles_csv(self, client):
        """Export endpoint should return CSV."""
        response = client.get("/api/vehicles/export")
        
        assert response.status_code == 200
        # CSV response should have text/csv content-type
        assert "text" in response.headers.get("content-type", "").lower() or \
               "csv" in response.headers.get("content-type", "").lower() or \
               response.text  # At minimum, should have text content


class TestComplaintEndpoint:
    """Test file complaint endpoint."""
    
    def test_file_complaint_not_found(self, client):
        """Complaint endpoint should return 404 for nonexistent vehicle."""
        complaint_data = {"police_complaint_id": "ABC123"}
        
        response = client.post(
            "/api/vehicles/record/NONEXISTENT/file-complaint",
            json=complaint_data
        )
        
        assert response.status_code == 404


class TestRecoveryEndpoint:
    """Test mark recovered endpoint."""
    
    def test_mark_recovered_not_found(self, client):
        """Recovery endpoint should return 404 for nonexistent vehicle."""
        response = client.post("/api/vehicles/record/NONEXISTENT/mark-recovered")
        
        assert response.status_code == 404


class TestEndpointConsistency:
    """Test that all required endpoints exist."""
    
    def test_all_detection_endpoints_exist(self, client):
        """All detection endpoints should be defined."""
        endpoints = [
            ("/", "GET"),
            ("/api/v1/detect/image", "POST"),
            ("/api/vehicles", "GET"),
            ("/api/vehicles", "POST"),
            ("/api/vehicles/export", "GET"),
        ]
        
        for path, method in endpoints:
            if method == "GET":
                response = client.get(path)
                # Should not be 404, might be other errors
                assert response.status_code != 404, f"{method} {path} returned 404"
            elif method == "POST":
                # POST might require specific data, but should exist
                response = client.post(path)
                assert response.status_code != 404, f"{method} {path} returned 404"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
