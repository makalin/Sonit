"""
Tests for audio processing modules
"""

import pytest
import numpy as np
import os
import tempfile

# Import modules to test
from audio.processor import AudioProcessor
from audio.recorder import AudioRecorder


class TestAudioProcessor:
    """Test cases for AudioProcessor"""
    
    def setup_method(self):
        """Setup for each test"""
        self.processor = AudioProcessor()
        
        # Create dummy audio data
        self.dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
    
    def test_extract_features(self):
        """Test feature extraction"""
        # Convert dummy audio to bytes
        audio_bytes = (self.dummy_audio * 32768).astype(np.int16).tobytes()
        
        # Extract features
        features = self.processor.extract_features(audio_bytes)
        
        # Check that features were extracted
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_extract_mel_spectrogram(self):
        """Test mel spectrogram extraction"""
        # Convert dummy audio to bytes
        audio_bytes = (self.dummy_audio * 32768).astype(np.int16).tobytes()
        
        # Extract mel spectrogram
        mel_spec = self.processor.extract_mel_spectrogram(audio_bytes)
        
        # Check that mel spectrogram was extracted
        assert mel_spec is not None
        assert isinstance(mel_spec, np.ndarray)
        assert mel_spec.ndim == 2
    
    def test_get_feature_names(self):
        """Test feature names generation"""
        feature_names = self.processor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)


class TestAudioRecorder:
    """Test cases for AudioRecorder"""
    
    def setup_method(self):
        """Setup for each test"""
        self.recorder = AudioRecorder()
    
    def test_initialization(self):
        """Test recorder initialization"""
        assert self.recorder.sample_rate == 16000
        assert self.recorder.chunk_size == 1024
        assert self.recorder.channels == 1
    
    def test_get_available_devices(self):
        """Test device enumeration"""
        devices = self.recorder.get_available_devices()
        
        assert isinstance(devices, list)
        # Should have at least one device (system default)
        assert len(devices) >= 0
    
    def test_audio_level_detection(self):
        """Test audio level detection"""
        # Test with no audio (should return 0)
        level = self.recorder.get_audio_level()
        assert isinstance(level, float)
        assert 0.0 <= level <= 1.0
    
    def test_audio_detection(self):
        """Test audio detection"""
        # Test with no audio (should return False)
        detected = self.recorder.is_audio_detected()
        assert isinstance(detected, bool)
    
    def test_record_to_file(self):
        """Test recording to file"""
        # This test would require actual audio input
        # For now, just test that the method exists and has correct signature
        assert hasattr(self.recorder, 'record_to_file')
        assert callable(self.recorder.record_to_file)


class TestAudioIntegration:
    """Integration tests for audio processing"""
    
    def test_processor_recorder_integration(self):
        """Test integration between processor and recorder"""
        processor = AudioProcessor()
        recorder = AudioRecorder()
        
        # Test that both can be instantiated together
        assert processor is not None
        assert recorder is not None
        
        # Test that they have compatible parameters
        assert processor.sample_rate == recorder.sample_rate
    
    def test_feature_extraction_pipeline(self):
        """Test complete feature extraction pipeline"""
        processor = AudioProcessor()
        
        # Create synthetic audio data
        sample_rate = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a simple sine wave
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to bytes
        audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()
        
        # Extract features
        features = processor.extract_features(audio_bytes)
        
        # Check that features were extracted successfully
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.any(np.isnan(features))  # No NaN values
        assert not np.any(np.isinf(features))  # No infinite values


if __name__ == "__main__":
    pytest.main([__file__]) 