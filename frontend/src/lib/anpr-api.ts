/**
 * ANPR API Types and Client
 * Handles communication with the FastAPI backend
 */

export interface VehicleMatchSummary {
  plate_number: string;
  status: 'Clear' | 'Stolen/Missing' | 'Recovered';
  vehicle_make: string;
  vehicle_model: string;
  owner_name: string;
  registered_rto_state: string;
  registered_rto_code: string;
  police_complaint_id?: string | null;
}

export interface PlateDetection {
  db_id?: number;
  plate_text: string;
  confidence: number;
  bbox: [number, number, number, number];
  vehicle_crop_url: string;
  plate_crop_url: string;
  raw_ocr_text?: string;
  ocr_engine?: string;
  top_ocr_candidates?: string[];
  format_score?: number;
  review_required?: boolean;
  registry_match?: VehicleMatchSummary | null;
  human_corrected_text?: string | null;
  human_verified?: boolean;
  seen_count?: number;
  first_seen_sec?: number | null;
  last_seen_sec?: number | null;
  source_frame?: number | null;
  source_file_name?: string | null;
}

export interface ANPRResult {
  job_id: string;
  status: 'processing' | 'completed' | 'error';
  input_type: 'image' | 'video';
  total_detections: number;
  processing_time: number;
  detections: PlateDetection[];
  output_file_url?: string;
  error?: string;
  progress?: number;
  stage?: string | null;
  alert_count?: number;
  review_count?: number;
}

const _apiHost = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000';
const API_BASE = `${_apiHost}/api`;
export const API_HOST = _apiHost;

type DetectImageResponse = {
  success: boolean;
  image_url?: string;
  detections?: Array<Partial<PlateDetection> & { db_id?: number }>;
};

export function getAssetUrl(path: string | undefined): string {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  return `${API_HOST}${path}`;
}

export async function uploadImages(files: File[]): Promise<ANPRResult> {
  if (files.length === 0) {
    throw new Error('No image files selected.');
  }

  const startedAt = performance.now();
  const detections: PlateDetection[] = [];

  for (const file of files) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_HOST}/api/v1/detect/image`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to upload image ${file.name}: ${response.statusText}`);
    }

    const payload = (await response.json()) as DetectImageResponse;
    if (!payload.success) {
      throw new Error(`Image processing failed for ${file.name}`);
    }

    const imageDetections = (payload.detections ?? []).map((det): PlateDetection => {
      const rawBbox = Array.isArray(det.bbox) ? det.bbox : [];
      const bbox: [number, number, number, number] = [
        Number(rawBbox[0] ?? 0),
        Number(rawBbox[1] ?? 0),
        Number(rawBbox[2] ?? 0),
        Number(rawBbox[3] ?? 0),
      ];

      return {
        db_id: typeof det.db_id === 'number' ? det.db_id : undefined,
        plate_text: det.plate_text ?? '',
        confidence: Number(det.confidence ?? 0),
        bbox,
        vehicle_crop_url: det.vehicle_crop_url ?? payload.image_url ?? '',
        plate_crop_url: det.plate_crop_url ?? payload.image_url ?? '',
        raw_ocr_text: det.raw_ocr_text,
        ocr_engine: "EasyOCR",
        format_score: typeof det.format_score === 'number' ? det.format_score : 0,
        review_required: (det.format_score ?? 0) < 0.8,
        registry_match: det.registry_match ?? null,
        human_corrected_text: det.human_corrected_text ?? null,
        human_verified: det.human_verified,
        seen_count: det.seen_count,
        first_seen_sec: det.first_seen_sec ?? null,
        last_seen_sec: det.last_seen_sec ?? null,
        source_frame: det.source_frame ?? null,
        source_file_name: det.source_file_name ?? file.name,
      };
    });

    detections.push(...imageDetections);
  }

  const processingTime = (performance.now() - startedAt) / 1000;
  const alertCount = detections.filter(d => d.registry_match?.status === 'Stolen/Missing').length;
  const reviewCount = detections.filter(d => d.review_required).length;

  return {
    job_id: `image-${Date.now()}`,
    status: 'completed',
    input_type: 'image',
    total_detections: detections.length,
    processing_time: processingTime,
    detections,
    alert_count: alertCount,
    review_count: reviewCount,
    progress: 100,
    stage: 'Completed',
  };
}

export async function correctDetection(recordId: number, correctedText: string): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`${API_HOST}/api/v1/detect/correct/${recordId}?corrected_text=${encodeURIComponent(correctedText)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`Failed to save correction: ${response.statusText}`);
  }

  return response.json();
}

export async function uploadVideo(file: File): Promise<ANPRResult> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(`${API_BASE}/process-video`, { method: 'POST', body: formData });
  if (!response.ok) throw new Error(`Failed to upload video: ${response.statusText}`);
  return response.json();
}

export async function getJobResults(jobId: string): Promise<ANPRResult> {
  const response = await fetch(`${API_BASE}/results/${jobId}`);
  if (!response.ok) throw new Error(`Failed to fetch results: ${response.statusText}`);
  return response.json();
}

export function exportResultsAsJSON(results: ANPRResult, filename?: string) {
  const json = JSON.stringify(results, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || `anpr-results-${results.job_id}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function exportResultsAsCSV(results: ANPRResult, filename?: string) {
  let csv = 'Detection #,License Plate,Corrected Plate,Confidence,Format Score,Registry Status,Bbox (x1, y1, x2, y2)\n';
  results.detections.forEach((det, idx) => {
    csv += `${idx + 1},"${det.plate_text}","${det.human_corrected_text ?? ''}",${(det.confidence * 100).toFixed(1)}%,${((det.format_score ?? 0) * 100).toFixed(1)}%,"${det.registry_match?.status ?? 'Unmatched'}","${det.bbox.join(', ')}"\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || `anpr-results-${results.job_id}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}