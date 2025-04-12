# WhisperX CUDA Runtime Configuration Plan

## Overview

This document outlines the plan to resolve WhisperX diarization issues by managing CUDA version compatibility through container runtime configuration, specifically addressing the mismatch between host CUDA 12.8 and container CUDA 12.6.3.

## Current Environment

* Host System:
  - CUDA Version: 12.8
  - NVIDIA Driver: 570.124.06
  - GPUs: 2x RTX 3060, 1x Quadro RTX 4000
  - Other CUDA Apps: Ollama (version-flexible)

* Container:
  - CUDA Version: 12.6.3
  - Base Image: python:3.11-slim
  - Current Error: `RuntimeError: Error in dlopen for library libnvrtc.so.12`

## Implementation Plan

### 1. CUDA Library Verification
```bash
# Check installed CUDA versions
ls -l /usr/local/cuda*

# Verify library locations
find /usr -name "libnvrtc*"

# Check CUDA installation paths
which nvcc

# Review library dependencies
ldd $(which nvidia-smi)
```

### 2. Container Runtime Configuration
```bash
docker run \
  --gpus all \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v /usr/local/cuda-12.6:/usr/local/cuda-12.6 \
  -v .:/app \
  whisperx:distil-large-v3 [command]
```

### 3. Library Management Scenarios

#### If CUDA 12.6 Libraries Present:
1. Mount appropriate library paths
2. Configure container environment variables:
   ```bash
   -e LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
   -e CUDA_HOME=/usr/local/cuda-12.6
   ```
3. Test diarization functionality

#### If CUDA 12.6 Libraries Missing:
1. Download required libraries:
   ```bash
   # Create directory for CUDA 12.6
   sudo mkdir -p /usr/local/cuda-12.6
   
   # Download and extract CUDA 12.6 libraries
   # (Specific commands to be determined based on NVIDIA's repository)
   ```
2. Configure mount points and access
3. Verify library compatibility

### 4. Testing Protocol

1. Basic GPU Access:
   ```bash
   nvidia-smi
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. Library Verification:
   ```bash
   python3 -c "import torch; print(torch.version.cuda)"
   ldconfig -p | grep cuda
   ```

3. Diarization Test:
   - Run with small audio sample
   - Monitor GPU utilization
   - Verify speaker detection
   - Check performance metrics

## Success Criteria

1. Container successfully accesses CUDA libraries
2. PyTorch recognizes correct CUDA version
3. Diarization completes without library errors
4. Performance remains within acceptable range

## Rollback Plan

If container runtime approach fails:
1. Document specific failure points
2. Consider host CUDA downgrade as fallback
3. Evaluate alternative diarization solutions

## Next Steps

1. Execute library verification steps
2. Implement container runtime configuration
3. Test and validate solution
4. Document successful configuration