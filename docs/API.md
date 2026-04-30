# PY-BRAIN API Documentation

## Overview

The PY-BRAIN API provides a REST interface for brain tumor segmentation and analysis. It supports:
- Case management (upload, retrieve, delete)
- Async job processing via Celery
- DICOM and NIfTI file handling
- Longitudinal comparison
- Audit logging for HIPAA compliance

**Base URL**: `http://localhost:8000`

**Interactive Documentation**: `http://localhost:8000/docs` (FastAPI auto-generated)

## Authentication

All endpoints require authentication via JWT bearer token or API key (research mode).

### Headers
```
Authorization: Bearer <token-or-api-key>
```

### Research Mode
In development/research mode, API keys are accepted. Configure via:
```bash
API_KEYS=key1,key2,key3
```

### Production Mode
In production, JWT tokens are required. Configure via:
```bash
JWT_SECRET=your-secret-key
```

## Endpoints

### Health

#### `GET /health`
Basic health check - always returns 200 if the API is running.

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

#### `GET /ready`
Readiness check - verifies Redis and PostgreSQL connectivity.

**Response** (healthy):
```json
{
  "status": "ready",
  "redis": "healthy",
  "database": "healthy"
}
```

**Response** (not ready):
```json
{
  "status": "not_ready",
  "redis": "unhealthy",
  "database": "healthy"
}
```

### Cases

#### `POST /cases`
Upload DICOM zip or NIfTI files and create a case.

**Request** (multipart/form-data):
- `files`: Uploaded file (DICOM zip or NIfTI files)
- `patient_name` (optional): Patient name
- `patient_age` (optional): Patient age
- `patient_sex` (optional): Patient sex
- `analysis_mode` (optional): Analysis mode (glioma, mets, auto)

**Response**:
```json
{
  "case_id": "uuid-string",
  "status": "pending",
  "analysis_mode": "auto"
}
```

#### `GET /cases/{case_id}`
Get case status and results.

**Response**:
```json
{
  "case_id": "uuid-string",
  "status": "completed",
  "analysis_mode": "glioma",
  "patient_name": "John Doe",
  "patient_age": 65,
  "patient_sex": "M",
  "volumes": {
    "wt_cc": 10.5,
    "tc_cc": 5.2,
    "et_cc": 3.1,
    "nc_cc": 5.3
  },
  "mets_result": null,
  "longitudinal_result": null,
  "error_message": null,
  "created_at": "2026-04-30T12:00:00",
  "updated_at": "2026-04-30T12:05:00"
}
```

#### `GET /cases/{case_id}/segmentation`
Download segmentation as NIfTI file.

**Response**: Binary NIfTI file (application/gzip)

#### `GET /cases/{case_id}/report`
Download PDF report.

**Response**: Binary PDF file (application/pdf)

#### `GET /cases/{case_id}/dicom-seg`
Stream DICOM-SEG file.

**Response**: Binary DICOM file (application/dicom)

#### `POST /cases/{case_id}/segment`
Trigger segmentation for a case.

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "celery_task_id": "celery-task-id"
}
```

#### `POST /cases/{case_id}/longitudinal/{prior_id}`
Trigger longitudinal comparison between current and prior case.

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "celery_task_id": "celery-task-id"
}
```

#### `DELETE /cases/{case_id}`
Soft delete a case (audit trail preserved).

**Response**:
```json
{
  "message": "Case soft deleted",
  "case_id": "uuid-string"
}
```

### Jobs

#### `POST /jobs/segment`
Enqueue a segmentation job for a case.

**Request body**:
```json
{
  "case_id": "uuid-string"
}
```

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "celery_task_id": "celery-task-id"
}
```

#### `GET /jobs/{job_id}`
Get job status and progress.

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "success",
  "progress": 100,
  "result": {
    "volumes": {
      "wt_cc": 10.5,
      "tc_cc": 5.2,
      "et_cc": 3.1,
      "nc_cc": 5.3
    },
    "segmentation_path": "cases/uuid/segmentation.nii.gz",
    "report_path": "cases/uuid/report.pdf"
  },
  "error_message": null,
  "created_at": "2026-04-30T12:00:00",
  "started_at": "2026-04-30T12:00:05",
  "completed_at": "2026-04-30T12:05:00"
}
```

**Job Statuses**:
- `pending`: Job queued, not started
- `started`: Job in progress
- `success`: Job completed successfully
- `failure`: Job failed

#### `POST /jobs/longitudinal`
Enqueue a longitudinal comparison job.

**Request body**:
```json
{
  "case_id": "uuid-string",
  "prior_case_id": "uuid-string"
}
```

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "celery_task_id": "celery-task-id"
}
```

#### `POST /jobs/export-dicom`
Enqueue a DICOM export job.

**Request body**:
```json
{
  "case_id": "uuid-string",
  "formats": ["dicom-seg", "dicom-sr"]
}
```

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "celery_task_id": "celery-task-id"
}
```

## Error Responses

All endpoints may return error responses:

### 401 Unauthorized
```json
{
  "detail": "Missing authentication credentials"
}
```

### 404 Not Found
```json
{
  "detail": "Case not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

## Rate Limiting

Currently not implemented. Add via `slowapi` or upstream proxy in production.

## CORS

Configure allowed origins via:
```bash
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

## OpenAPI Schema

The full OpenAPI schema is available at:
- JSON: `http://localhost:8000/openapi.json`
- UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Example Usage

### Upload and Segment a Case

```bash
# Upload case
curl -X POST http://localhost:8000/cases \
  -H "Authorization: Bearer dev-api-key-123" \
  -F "files=@case.zip" \
  -F "analysis_mode=auto"

# Response: {"case_id": "...", "status": "pending"}

# Trigger segmentation
curl -X POST http://localhost:8000/cases/{case_id}/segment \
  -H "Authorization: Bearer dev-api-key-123"

# Response: {"job_id": "...", "status": "pending"}

# Poll job status
curl http://localhost:8000/jobs/{job_id} \
  -H "Authorization: Bearer dev-api-key-123"

# Response: {"job_id": "...", "status": "success", ...}

# Get case results
curl http://localhost:8000/cases/{case_id} \
  -H "Authorization: Bearer dev-api-key-123"

# Download segmentation
curl http://localhost:8000/cases/{case_id}/segmentation \
  -H "Authorization: Bearer dev-api-key-123" \
  -o segmentation.nii.gz
```

## Audit Logging

All patient data access and modifications are logged for HIPAA compliance:
- Every read/write of patient data is logged
- Every API call is logged with user, timestamp, IP, action, resource
- Audit trail is preserved even on soft delete

Audit logs can be retrieved via the database (admin endpoint to be added).
