import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Optional, Union
import torch
import logging
from contextlib import contextmanager

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.types import TranscriptionResult, AlignedTranscriptionResult

logger = logging.getLogger(__name__)

@contextmanager
def gpu_memory_manager():
    """Context manager for handling GPU memory operations safely"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def check_gpu_availability():
    """Check if CUDA GPU is available and properly configured"""
    if not torch.cuda.is_available():
        logger.warning("CUDA GPU is not available. Using CPU instead.")
        return False, "cpu"
    
    try:
        # Test CUDA functionality
        test_tensor = torch.tensor([1.0], device="cuda")
        del test_tensor
        logger.info(f"CUDA GPU available: {torch.cuda.get_device_name()}")
        return True, "cuda"
    except RuntimeError as e:
        logger.error(f"CUDA GPU error: {str(e)}")
        return False, "cpu"

class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        # Check GPU availability and set device
        gpu_available, default_device = check_gpu_availability()
        if device is None:
            device = default_device
        
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        try:
            # Initialize pipeline with error handling
            self.model = Pipeline.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
        except Exception as e:
            logger.error(f"Failed to initialize PyAnnote pipeline: {str(e)}")
            raise
        
        # Configure pipeline parameters for better speaker detection
        self.model.instantiate({
            "segmentation": {
                "min_duration_on": 0.1,      # Shorter minimum speech duration
                "threshold": 0.45,           # Lower threshold for better detection
                "min_duration_off": 0.1      # Shorter silence duration
            },
            "clustering": {
                "method": "centroid",        # Use centroid clustering
                "min_cluster_size": 2,       # Minimum segments per speaker
                "threshold": 0.65,           # Lower threshold for more distinct clusters
                "max_gap": 3.0,             # Maximum gap between segments
                "min_duration": 0.1          # Minimum duration per speaker
            }
        })

        # Move to specified device with error handling
        try:
            if device.type == "cuda":
                with gpu_memory_manager():
                    self.model = self.model.to(device)
                logger.info(f"Successfully moved model to {device}")
        except RuntimeError as e:
            logger.error(f"Failed to move model to {device}: {str(e)}")
            if device.type == "cuda":
                logger.warning("Falling back to CPU")
                self.device = torch.device("cpu")
                self.model = self.model.to("cpu")

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        # Load audio if path provided
        if isinstance(audio, str):
            audio = load_audio(audio)
        
        # Convert to tensor and move to device
        waveform = torch.from_numpy(audio[None, :]).to(self.device)
        audio_data = {
            'waveform': waveform,
            'sample_rate': SAMPLE_RATE
        }
        
        # Process with GPU memory optimization and error handling
        try:
            with gpu_memory_manager(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                segments = self.model(
                    audio_data,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("GPU out of memory error. Try reducing batch size or using CPU")
                torch.cuda.empty_cache()
            raise RuntimeError(f"Error during diarization: {str(e)}")

        # Convert to DataFrame and merge close segments
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        
        # Merge segments from same speaker that are close together
        diarize_df = self.merge_segments(diarize_df)
        
        return diarize_df

    def merge_segments(self, df, gap_threshold=0.5):
        """Merge segments from the same speaker that are close together"""
        df = df.sort_values('start')
        merged = []
        
        current = None
        for _, row in df.iterrows():
            if current is None:
                current = row
                continue
                
            if (row['speaker'] == current['speaker'] and 
                row['start'] - current['end'] < gap_threshold):
                # Merge segments
                current['end'] = row['end']
                current['segment'] = current['segment'].union(row['segment'])
            else:
                merged.append(current)
                current = row
                
        if current is not None:
            merged.append(current)
            
        return pd.DataFrame(merged)

def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    fill_nearest=True,
) -> dict:
    transcript_segments = transcript_result["segments"]
    
    # Pre-process diarization segments
    diarize_df = diarize_df.sort_values('start')
    
    # Create time windows for each speaker
    speaker_windows = []
    for _, row in diarize_df.iterrows():
        speaker_windows.append({
            'start': row['start'],
            'end': row['end'],
            'speaker': row['speaker']
        })
    
    for seg in transcript_segments:
        # Find overlapping speaker windows
        seg_start = seg['start']
        seg_end = seg['end']
        
        overlapping_speakers = {}
        total_overlap = 0
        
        for window in speaker_windows:
            overlap_start = max(seg_start, window['start'])
            overlap_end = min(seg_end, window['end'])
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                total_overlap += overlap_duration
                overlapping_speakers[window['speaker']] = overlapping_speakers.get(window['speaker'], 0) + overlap_duration
        
        # Assign the speaker with maximum overlap
        if overlapping_speakers:
            seg["speaker"] = max(overlapping_speakers.items(), key=lambda x: x[1])[0]
        elif fill_nearest:
            # If no overlap, find nearest speaker
            distances = []
            seg_mid = (seg_start + seg_end) / 2
            
            for window in speaker_windows:
                window_mid = (window['start'] + window['end']) / 2
                distances.append((abs(seg_mid - window_mid), window['speaker']))
            
            seg["speaker"] = min(distances, key=lambda x: x[0])[1]
        
        # Assign same speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    word["speaker"] = seg["speaker"]
    
    return transcript_result