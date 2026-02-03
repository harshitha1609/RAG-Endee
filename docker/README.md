# Docker Setup for Endee RAG System

## Overview
This directory contains Docker Compose configuration for running the Endee vector database service.

## Prerequisites
- Docker and Docker Compose installed
- Access to Endee/ndd Docker image (update image name in docker-compose.yml as needed)

## Usage

### Start Endee Service
```bash
# From project root directory
docker-compose -f docker/docker-compose.yml up -d
```

### Check Service Status
```bash
docker-compose -f docker/docker-compose.yml ps
```

### View Logs
```bash
docker-compose -f docker/docker-compose.yml logs -f endee
```

### Stop Service
```bash
docker-compose -f docker/docker-compose.yml down
```

### Health Check
Once running, Endee should be accessible at:
- **Base URL:** http://localhost:8080
- **Health Check:** http://localhost:8080/health

## Configuration

### Environment Variables
- `ENDEE_LOG_LEVEL`: Logging level (default: INFO)
- `ENDEE_PORT`: Service port (default: 8080)

### Network
- Service runs on `rag-network` bridge network
- Port 8080 is exposed to host

### Health Check
- Endpoint: `/health`
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3
- Start period: 40 seconds

## Troubleshooting

### Image Not Found
If you get "image not found" errors:
1. Update the image name in `docker-compose.yml` to the correct Endee image
2. Or build locally if you have Endee source code
3. Or use alternative vector database for testing

### Port Conflicts
If port 8080 is already in use:
1. Change the port mapping in `docker-compose.yml`
2. Update backend configuration to match new port

### Health Check Failures
- Ensure the health endpoint path is correct for your Endee version
- Adjust timeout and retry values if needed
- Check Endee logs for startup issues