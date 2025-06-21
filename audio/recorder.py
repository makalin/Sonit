"""
Audio recorder module for Sonit
Handles real-time audio capture from microphone
"""

import pyaudio
import numpy as np
import wave
import os
import threading
import time
from contextlib import contextmanager


class AudioRecorder:
    """Real-time audio recorder for capturing vocal gestures"""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1, format_type=pyaudio.paInt16):
        """
        Initialize audio recorder
        
        Args:
            sample_rate (int): Audio sample rate in Hz
            chunk_size (int): Number of frames per buffer
            channels (int): Number of audio channels (1 for mono, 2 for stereo)
            format_type: PyAudio format type
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format_type = format_type
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # Audio buffer
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Recording settings
        self.recording_path = 'data/recordings'
        os.makedirs(self.recording_path, exist_ok=True)
    
    def __del__(self):
        """Cleanup PyAudio resources"""
        self.stop_stream()
        if hasattr(self, 'audio'):
            self.audio.terminate()
    
    @contextmanager
    def start_stream(self):
        """Context manager for starting and stopping audio stream"""
        try:
            self._start_stream()
            yield self
        finally:
            self._stop_stream()
    
    def _start_stream(self):
        """Start the audio stream"""
        if self.stream is not None:
            return
        
        try:
            self.stream = self.audio.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            self.is_recording = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to start audio stream: {e}")
    
    def _stop_stream(self):
        """Stop the audio stream"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            self.is_recording = False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        with self.buffer_lock:
            self.audio_buffer.append(in_data)
            
            # Keep only last 10 seconds of audio
            max_buffers = int(10 * self.sample_rate / self.chunk_size)
            if len(self.audio_buffer) > max_buffers:
                self.audio_buffer.pop(0)
        
        return (in_data, pyaudio.paContinue)
    
    def get_audio_chunk(self):
        """Get the latest audio chunk"""
        with self.buffer_lock:
            if self.audio_buffer:
                return self.audio_buffer[-1]
            return None
    
    def get_audio_data(self, duration_seconds=2.0):
        """
        Get audio data for a specified duration
        
        Args:
            duration_seconds (float): Duration to record in seconds
            
        Returns:
            bytes: Audio data
        """
        if not self.is_recording:
            raise RuntimeError("Audio stream not started")
        
        # Calculate number of chunks needed
        chunks_needed = int(duration_seconds * self.sample_rate / self.chunk_size)
        chunks = []
        
        start_time = time.time()
        while len(chunks) < chunks_needed and (time.time() - start_time) < duration_seconds + 1.0:
            with self.buffer_lock:
                if self.audio_buffer:
                    chunks.append(self.audio_buffer[-1])
            time.sleep(0.01)  # Small delay to prevent busy waiting
        
        if chunks:
            return b''.join(chunks)
        return None
    
    def record_to_file(self, filename, duration_seconds=2.0):
        """
        Record audio and save to file
        
        Args:
            filename (str): Output filename
            duration_seconds (float): Recording duration in seconds
            
        Returns:
            str: Path to saved file
        """
        audio_data = self.get_audio_data(duration_seconds)
        
        if audio_data is None:
            raise RuntimeError("No audio data captured")
        
        # Ensure filename has .wav extension
        if not filename.endswith('.wav'):
            filename += '.wav'
        
        filepath = os.path.join(self.recording_path, filename)
        
        # Save as WAV file
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format_type))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        
        return filepath
    
    def get_audio_level(self):
        """
        Get current audio level (RMS)
        
        Returns:
            float: RMS audio level (0.0 to 1.0)
        """
        audio_chunk = self.get_audio_chunk()
        
        if audio_chunk is None:
            return 0.0
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_array**2))
        
        # Normalize to 0-1 range (assuming 16-bit audio)
        normalized_rms = rms / 32768.0
        
        return min(normalized_rms, 1.0)
    
    def is_audio_detected(self, threshold=0.01):
        """
        Check if audio is detected above threshold
        
        Args:
            threshold (float): Audio level threshold (0.0 to 1.0)
            
        Returns:
            bool: True if audio detected
        """
        return self.get_audio_level() > threshold
    
    def get_available_devices(self):
        """
        Get list of available audio input devices
        
        Returns:
            list: List of device info dictionaries
        """
        devices = []
        
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            
            # Only include input devices
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                })
        
        return devices
    
    def set_device(self, device_index):
        """
        Set audio input device
        
        Args:
            device_index (int): Device index from get_available_devices()
        """
        if self.is_recording:
            raise RuntimeError("Cannot change device while recording")
        
        # Verify device exists
        device_info = self.audio.get_device_info_by_index(device_index)
        if device_info['maxInputChannels'] == 0:
            raise ValueError(f"Device {device_index} is not an input device")
        
        # Update stream parameters (will be used on next start_stream)
        self.device_index = device_index
    
    def get_current_device_info(self):
        """
        Get current device information
        
        Returns:
            dict: Device info or None if no device set
        """
        if hasattr(self, 'device_index'):
            return self.audio.get_device_info_by_index(self.device_index)
        return None 