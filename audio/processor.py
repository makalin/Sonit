"""
Audio processor module for Sonit
Extracts features from audio data for machine learning
"""

import numpy as np
import librosa
import librosa.feature
import librosa.util
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


class AudioProcessor:
    """Audio feature extraction for vocal gesture analysis"""
    
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        Initialize audio processor
        
        Args:
            sample_rate (int): Audio sample rate in Hz
            n_mfcc (int): Number of MFCC coefficients
            n_fft (int): FFT window size
            hop_length (int): Number of samples between frames
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Feature extraction parameters
        self.n_mels = 128
        self.n_chroma = 12
        self.n_spectral = 7
        
        # Preprocessing parameters
        self.pre_emphasis = 0.97
        self.frame_length = 0.025  # 25ms
        self.frame_step = 0.010    # 10ms
    
    def extract_features(self, audio_data):
        """
        Extract comprehensive features from audio data
        
        Args:
            audio_data (bytes): Raw audio data
            
        Returns:
            numpy.ndarray: Feature vector or None if processing fails
        """
        try:
            # Convert bytes to numpy array
            audio_array = self._bytes_to_array(audio_data)
            
            if audio_array is None or len(audio_array) == 0:
                return None
            
            # Preprocess audio
            audio_processed = self._preprocess_audio(audio_array)
            
            # Extract various feature types
            features = []
            
            # MFCC features
            mfcc_features = self._extract_mfcc(audio_processed)
            if mfcc_features is not None:
                features.extend(mfcc_features)
            
            # Spectral features
            spectral_features = self._extract_spectral_features(audio_processed)
            if spectral_features is not None:
                features.extend(spectral_features)
            
            # Chroma features
            chroma_features = self._extract_chroma_features(audio_processed)
            if chroma_features is not None:
                features.extend(chroma_features)
            
            # Temporal features
            temporal_features = self._extract_temporal_features(audio_processed)
            if temporal_features is not None:
                features.extend(temporal_features)
            
            # Statistical features
            statistical_features = self._extract_statistical_features(audio_processed)
            if statistical_features is not None:
                features.extend(statistical_features)
            
            # Convert to numpy array and normalize
            if features:
                feature_vector = np.array(features, dtype=np.float32)
                return self._normalize_features(feature_vector)
            
            return None
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _bytes_to_array(self, audio_data):
        """Convert audio bytes to numpy array"""
        try:
            # Assume 16-bit PCM audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1, 1] range
            audio_array = audio_array / 32768.0
            
            return audio_array
            
        except Exception as e:
            print(f"Error converting audio bytes: {e}")
            return None
    
    def _preprocess_audio(self, audio_array):
        """Preprocess audio signal"""
        try:
            # Apply pre-emphasis filter
            audio_emphasized = librosa.effects.preemphasis(audio_array, coef=self.pre_emphasis)
            
            # Remove silence
            audio_trimmed, _ = librosa.effects.trim(audio_emphasized, top_db=20)
            
            # Pad if too short
            min_length = int(0.1 * self.sample_rate)  # Minimum 100ms
            if len(audio_trimmed) < min_length:
                audio_trimmed = np.pad(audio_trimmed, (0, min_length - len(audio_trimmed)), 'constant')
            
            return audio_trimmed
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return audio_array
    
    def _extract_mfcc(self, audio_array):
        """Extract MFCC features"""
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_array,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Calculate statistics over time
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
            mfcc_delta2 = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
            
            # Combine all MFCC features
            mfcc_features = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta, mfcc_delta2])
            
            return mfcc_features
            
        except Exception as e:
            print(f"Error extracting MFCC features: {e}")
            return None
    
    def _extract_spectral_features(self, audio_array):
        """Extract spectral features"""
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_array,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_array,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_array,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio_array,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y=audio_array,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Combine spectral features
            spectral_features = [
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_contrast), np.std(spectral_contrast),
                np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
            ]
            
            return np.array(spectral_features)
            
        except Exception as e:
            print(f"Error extracting spectral features: {e}")
            return None
    
    def _extract_chroma_features(self, audio_array):
        """Extract chroma features"""
        try:
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio_array,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Chroma statistics
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            chroma_max = np.max(chroma, axis=1)
            
            # Combine chroma features
            chroma_features = np.concatenate([chroma_mean, chroma_std, chroma_max])
            
            return chroma_features
            
        except Exception as e:
            print(f"Error extracting chroma features: {e}")
            return None
    
    def _extract_temporal_features(self, audio_array):
        """Extract temporal features"""
        try:
            # RMS energy
            rms = librosa.feature.rms(
                y=audio_array,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Energy
            energy = np.sum(audio_array**2)
            
            # Duration
            duration = len(audio_array) / self.sample_rate
            
            # Tempo (if possible)
            try:
                tempo, _ = librosa.beat.beat_track(
                    y=audio_array,
                    sr=self.sample_rate,
                    hop_length=self.hop_length
                )
            except:
                tempo = 0.0
            
            # Combine temporal features
            temporal_features = [
                np.mean(rms), np.std(rms),
                energy,
                duration,
                tempo
            ]
            
            return np.array(temporal_features)
            
        except Exception as e:
            print(f"Error extracting temporal features: {e}")
            return None
    
    def _extract_statistical_features(self, audio_array):
        """Extract statistical features"""
        try:
            # Basic statistics
            mean = np.mean(audio_array)
            std = np.std(audio_array)
            var = np.var(audio_array)
            skewness = skew(audio_array)
            kurt = kurtosis(audio_array)
            
            # Percentiles
            percentiles = np.percentile(audio_array, [10, 25, 50, 75, 90])
            
            # Range
            audio_range = np.max(audio_array) - np.min(audio_array)
            
            # Combine statistical features
            statistical_features = [
                mean, std, var, skewness, kurt,
                *percentiles,
                audio_range
            ]
            
            return np.array(statistical_features)
            
        except Exception as e:
            print(f"Error extracting statistical features: {e}")
            return None
    
    def _normalize_features(self, feature_vector):
        """Normalize feature vector"""
        try:
            # Replace infinite values with 0
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Z-score normalization
            mean = np.mean(feature_vector)
            std = np.std(feature_vector)
            
            if std > 0:
                feature_vector = (feature_vector - mean) / std
            else:
                feature_vector = feature_vector - mean
            
            return feature_vector
            
        except Exception as e:
            print(f"Error normalizing features: {e}")
            return feature_vector
    
    def extract_mel_spectrogram(self, audio_data, normalize=True):
        """
        Extract mel spectrogram for visualization
        
        Args:
            audio_data (bytes): Raw audio data
            normalize (bool): Whether to normalize the spectrogram
            
        Returns:
            numpy.ndarray: Mel spectrogram or None if processing fails
        """
        try:
            audio_array = self._bytes_to_array(audio_data)
            if audio_array is None:
                return None
            
            audio_processed = self._preprocess_audio(audio_array)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_processed,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            if normalize:
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec
            
        except Exception as e:
            print(f"Error extracting mel spectrogram: {e}")
            return None
    
    def get_feature_names(self):
        """Get list of feature names"""
        feature_names = []
        
        # MFCC features
        for i in range(self.n_mfcc):
            feature_names.append(f"mfcc_mean_{i}")
        for i in range(self.n_mfcc):
            feature_names.append(f"mfcc_std_{i}")
        for i in range(self.n_mfcc):
            feature_names.append(f"mfcc_delta_{i}")
        for i in range(self.n_mfcc):
            feature_names.append(f"mfcc_delta2_{i}")
        
        # Spectral features
        spectral_names = [
            "spectral_centroid_mean", "spectral_centroid_std",
            "spectral_bandwidth_mean", "spectral_bandwidth_std",
            "spectral_rolloff_mean", "spectral_rolloff_std",
            "spectral_contrast_mean", "spectral_contrast_std",
            "zero_crossing_rate_mean", "zero_crossing_rate_std"
        ]
        feature_names.extend(spectral_names)
        
        # Chroma features
        for i in range(self.n_chroma):
            feature_names.append(f"chroma_mean_{i}")
        for i in range(self.n_chroma):
            feature_names.append(f"chroma_std_{i}")
        for i in range(self.n_chroma):
            feature_names.append(f"chroma_max_{i}")
        
        # Temporal features
        temporal_names = [
            "rms_mean", "rms_std", "energy", "duration", "tempo"
        ]
        feature_names.extend(temporal_names)
        
        # Statistical features
        statistical_names = [
            "mean", "std", "variance", "skewness", "kurtosis",
            "percentile_10", "percentile_25", "percentile_50", "percentile_75", "percentile_90",
            "range"
        ]
        feature_names.extend(statistical_names)
        
        return feature_names 