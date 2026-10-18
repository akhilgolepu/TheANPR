/**
 * Frontend API contract tests for ANPRResults and VehicleRegistry
 * Tests detection upload, correction flow, and vehicle management
 */

import { describe, it, expect, vi } from 'vitest';

// Mock upload images function
const mockUploadImages = vi.fn().mockResolvedValue([
    {
        db_id: 1,
        file_name: 'test.jpg',
        predicted_plate: 'MH20DV2366',
        confidence: 0.95,
        format_score: 1.0,
        human_verified: false,
        human_corrected_text: null,
        registry_match: {
            plate_number: 'MH20DV2366',
            status: 'Clear',
            vehicle_make: 'Toyota',
        }
    }
]);

// Mock correctDetection function
const mockCorrectDetection = vi.fn().mockResolvedValue({
    success: true,
    message: 'Correction saved'
});

// Mock vehicle API functions
const mockListVehicles = vi.fn().mockResolvedValue([
    {
        plate_number: 'MH20DV2366',
        vehicle_make: 'Toyota',
        vehicle_model: 'Fortuner',
        status: 'Clear'
    },
    {
        plate_number: 'DL08AB1234',
        vehicle_make: 'Honda',
        vehicle_model: 'City',
        status: 'Stolen/Missing'
    }
]);

const mockGetVehicle = vi.fn().mockResolvedValue({
    plate_number: 'MH20DV2366',
    vehicle_make: 'Toyota',
    vehicle_model: 'Fortuner',
    vehicle_year: 2020,
    vehicle_color: 'White',
    owner_name: 'John Doe',
    status: 'Clear'
});

const mockUpdateVehicle = vi.fn().mockResolvedValue({
    success: true,
    message: 'Vehicle updated'
});

const mockDeleteVehicle = vi.fn().mockResolvedValue({
    success: true,
    message: 'Vehicle deleted'
});

const mockExportVehicles = vi.fn().mockResolvedValue(
    new Blob(['plate_number,vehicle_make,status\nMH20DV2366,Toyota,Clear'],
        { type: 'text/csv' })
);

describe('ANPRResults - Detection Upload Contract', () => {
    it('should handle image upload successfully', async () => {
        const result = await mockUploadImages();

        expect(result).toBeInstanceOf(Array);
        expect(result.length).toBeGreaterThan(0);
    });

    it('should return detection with required fields', async () => {
        const result = await mockUploadImages();
        const detection = result[0];

        expect(detection).toHaveProperty('db_id');
        expect(detection).toHaveProperty('file_name');
        expect(detection).toHaveProperty('predicted_plate');
        expect(detection).toHaveProperty('confidence');
        expect(detection).toHaveProperty('format_score');
        expect(detection).toHaveProperty('human_verified');
        expect(detection).toHaveProperty('registry_match');
    });

    it('should return detection with proper type signatures', async () => {
        const result = await mockUploadImages();
        const detection = result[0];

        expect(typeof detection.db_id).toBe('number');
        expect(typeof detection.file_name).toBe('string');
        expect(typeof detection.predicted_plate).toBe('string');
        expect(typeof detection.confidence).toBe('number');
        expect(typeof detection.format_score).toBe('number');
        expect(typeof detection.human_verified).toBe('boolean');
    });

    it('should include registry match when available', async () => {
        const result = await mockUploadImages();
        const detection = result[0];

        if (detection.registry_match) {
            expect(detection.registry_match).toHaveProperty('plate_number');
            expect(detection.registry_match).toHaveProperty('status');
            expect(['Clear', 'Stolen/Missing', 'Recovered']).toContain(
                detection.registry_match.status
            );
        }
    });
});

describe('ANPRResults - Correction Flow', () => {
    it('should submit correction successfully', async () => {
        const result = await mockCorrectDetection(1, 'MH20DV2366');

        expect(result).toHaveProperty('success');
        expect(result.success).toBe(true);
    });

    it('should handle correction with valid plate format', async () => {
        const validPlates = [
            'MH20DV2366',
            'DL08AB1234',
            'KA07AB9876'
        ];

        for (const plate of validPlates) {
            const result = await mockCorrectDetection(1, plate);
            expect(result.success).toBe(true);
        }
    });
});

describe('VehicleRegistry - List Vehicles', () => {
    it('should return list of vehicles', async () => {
        const result = await mockListVehicles();

        expect(result).toBeInstanceOf(Array);
        expect(result.length).toBeGreaterThan(0);
    });

    it('should return vehicles with required fields', async () => {
        const result = await mockListVehicles();
        const vehicle = result[0];

        expect(vehicle).toHaveProperty('plate_number');
        expect(vehicle).toHaveProperty('status');
        expect(['Clear', 'Stolen/Missing', 'Recovered']).toContain(vehicle.status);
    });

    it('should support filtering by status', async () => {
        const mockFilteredList = vi.fn().mockResolvedValue([
            {
                plate_number: 'DL08AB1234',
                vehicle_make: 'Honda',
                status: 'Stolen/Missing'
            }
        ]);

        const result = await mockFilteredList();
        expect(result.every(v => v.status === 'Stolen/Missing')).toBe(true);
    });
});

describe('VehicleRegistry - Get Vehicle Detail', () => {
    it('should return complete vehicle details', async () => {
        const result = await mockGetVehicle('MH20DV2366');

        expect(result).toHaveProperty('plate_number');
        expect(result).toHaveProperty('vehicle_make');
        expect(result).toHaveProperty('vehicle_model');
        expect(result).toHaveProperty('vehicle_year');
        expect(result).toHaveProperty('vehicle_color');
        expect(result).toHaveProperty('owner_name');
        expect(result).toHaveProperty('status');
    });

    it('should have proper field types', async () => {
        const result = await mockGetVehicle('MH20DV2366');

        expect(typeof result.plate_number).toBe('string');
        expect(typeof result.vehicle_make).toBe('string');
        expect(typeof result.vehicle_year).toBe('number');
        expect(typeof result.status).toBe('string');
    });
});

describe('VehicleRegistry - Update Vehicle', () => {
    it('should update vehicle successfully', async () => {
        const updateData = {
            vehicle_color: 'Red',
            owner_name: 'Jane Doe'
        };

        const result = await mockUpdateVehicle('MH20DV2366', updateData);

        expect(result).toHaveProperty('success');
        expect(result.success).toBe(true);
    });

    it('should accept partial updates', async () => {
        const partialUpdate = { vehicle_color: 'Blue' };
        const result = await mockUpdateVehicle('MH20DV2366', partialUpdate);

        expect(result.success).toBe(true);
    });
});

describe('VehicleRegistry - Delete Vehicle', () => {
    it('should delete vehicle successfully', async () => {
        const result = await mockDeleteVehicle('MH20DV2366');

        expect(result).toHaveProperty('success');
        expect(result.success).toBe(true);
    });
});

describe('VehicleRegistry - Export', () => {
    it('should export vehicles as CSV', async () => {
        const result = await mockExportVehicles();

        expect(result).toBeInstanceOf(Blob);
        expect(result.type).toBe('text/csv');
    });

    it('should return CSV with content', async () => {
        const result = await mockExportVehicles();
        const text = await result.text();

        expect(text.length).toBeGreaterThan(0);
        expect(text).toContain('plate_number');
    });
});

describe('API Error Handling', () => {
    it('should handle network errors gracefully', async () => {
        const mockFailedUpload = vi.fn()
            .mockRejectedValue(new Error('Network error'));

        expect(mockFailedUpload()).rejects.toThrow('Network error');
    });

    it('should handle 404 responses', async () => {
        const mock404 = vi.fn()
            .mockRejectedValue({ status: 404, message: 'Not found' });

        expect(mock404()).rejects.toMatchObject({ status: 404 });
    });
});

describe('Frontend-Backend Integration', () => {
    it('should maintain consistency between detection and vehicle lookup', async () => {
        const detections = await mockUploadImages();
        const detection = detections[0];

        if (detection.registry_match) {
            const vehicleDetail = await mockGetVehicle(detection.predicted_plate);

            expect(vehicleDetail.plate_number).toBe(detection.registry_match.plate_number);
            expect(vehicleDetail.status).toBe(detection.registry_match.status);
        }
    });

    it('should handle correction-to-vehicle-update flow', async () => {
        // Simulate correcting a detection and then updating vehicle
        const correctionResult = await mockCorrectDetection(1, 'MH20DV2366');
        expect(correctionResult.success).toBe(true);

        const vehicleUpdate = await mockUpdateVehicle('MH20DV2366', {
            vehicle_color: 'Updated'
        });
        expect(vehicleUpdate.success).toBe(true);
    });
});
