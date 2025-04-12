# GPU Diarization Version Compatibility Guide

## Version Matrix

### Core Components
| Component | Verified Version | Safe Range | Notes |
|-----------|-----------------|------------|--------|
| PyTorch | 2.0.1+cu117 | 2.0.0-2.1.x | CUDA 11.7 support required |
| PyAnnote | 3.1.0 | 3.1.0-3.1.x | GPU acceleration verified |
| CUDA | 11.7.1 | 11.7.x | Base container version |
| Host CUDA | 12.6 | 12.x | Backward compatible |

## Compatibility Rationale

### Testing Methodology
- Extensive testing performed with PyTorch 2.0.1+cu117
- PyAnnote 3.1.0 selected for proven GPU acceleration support
- CUDA 11.7.1 chosen for optimal compatibility with both components

### Performance Validation
- GPU acceleration confirmed functional
- Memory utilization optimized
- Processing speed significantly improved over CPU-only

## Version Mismatch Handling

### Warning Types and Resolution
1. **Training Version Mismatch**
   - Warning: "Model was trained with different version"
   - Severity: Non-critical
   - Action: Can be safely ignored

2. **TF32 Warnings**
   - Source: PyTorch
   - Impact: Documentation only
   - Resolution: No action needed

3. **CUDA Version Conflicts**
   - Symptom: GPU utilization issues
   - Resolution: Ensure CUDA 11.7.1 base image

## Configuration Recommendations

### constraints.txt
```
torch==2.0.1+cu117
pyannote.audio==3.1.0
```

### Dockerfile Configuration
```dockerfile
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

ENV CUDA_LAUNCH_BLOCKING=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Environment Variables
```bash
CUDA_LAUNCH_BLOCKING=1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Troubleshooting Guide

### Common Issues

1. **GPU Memory Allocation**
   - Error: CUDA out of memory
   - Solution: Adjust PYTORCH_CUDA_ALLOC_CONF
   - Reference: Issue #127 (memory optimization)

2. **Pipeline Initialization**
   - Error: Failed to move pipeline to GPU
   - Solution: Verify PyTorch CUDA availability
   - Reference: Issue #143 (GPU setup)

3. **Version Compatibility**
   - Error: Package conflicts
   - Solution: Strict adherence to version matrix
   - Reference: Issue #156 (dependency resolution)

## References

### GitHub Issues
- #127: Memory optimization for large files
- #143: GPU pipeline initialization
- #156: Package dependency resolution
- #162: CUDA version compatibility

### Documentation Links
- PyAnnote 3.1.0 Release Notes
- PyTorch 2.0.1 CUDA Support
- NVIDIA CUDA 11.7.1 Compatibility

## Monitoring and Maintenance

### Regular Checks
1. Version compatibility verification
2. GPU utilization monitoring
3. Memory usage optimization
4. Performance benchmarking

### Update Procedures
1. Test new versions in isolation
2. Verify GPU acceleration
3. Document compatibility changes
4. Update version matrices