"""
Main screen for Sonit application
Handles real-time sound capture and translation
"""

import os
import threading
import time
from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.properties import StringProperty, BooleanProperty, NumericProperty
from kivy.lang import Builder

# Import audio and model components
from audio.recorder import AudioRecorder
from audio.processor import AudioProcessor
from model.predictor import SoundPredictor
from utils.database import DatabaseManager


class MainScreen(Screen):
    """Main screen for real-time sound translation"""
    
    # Properties for UI updates
    status_text = StringProperty("Ready to listen...")
    confidence = NumericProperty(0.0)
    is_recording = BooleanProperty(False)
    current_translation = StringProperty("")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recorder = None
        self.processor = None
        self.predictor = None
        self.db_manager = None
        self.recording_thread = None
        self.should_record = False
        
        # Initialize components
        self._init_components()
        self._setup_ui()
    
    def _init_components(self):
        """Initialize audio and model components"""
        try:
            self.recorder = AudioRecorder()
            self.processor = AudioProcessor()
            self.predictor = SoundPredictor()
            self.db_manager = DatabaseManager()
            
            # Load user's trained model if available
            model_path = os.path.join('data', 'models', 'user_model.pth')
            if os.path.exists(model_path):
                self.predictor.load_model(model_path)
                self.status_text = "Model loaded successfully"
            else:
                self.status_text = "No trained model found. Please train first."
                
        except Exception as e:
            self.status_text = f"Error initializing: {str(e)}"
    
    def _setup_ui(self):
        """Setup the user interface"""
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        # Title
        title = Label(
            text="Sonit - Translating the Unspoken",
            size_hint_y=None,
            height=50,
            font_size='24sp',
            bold=True
        )
        layout.add_widget(title)
        
        # Status display
        self.status_label = Label(
            text=self.status_text,
            size_hint_y=None,
            height=40,
            font_size='16sp'
        )
        layout.add_widget(self.status_label)
        
        # Confidence bar
        confidence_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
        confidence_layout.add_widget(Label(text="Confidence:", size_hint_x=0.3))
        self.confidence_bar = ProgressBar(max=1.0, size_hint_x=0.7)
        confidence_layout.add_widget(self.confidence_bar)
        layout.add_widget(confidence_layout)
        
        # Translation display
        self.translation_label = Label(
            text="Translation will appear here...",
            size_hint_y=None,
            height=100,
            font_size='18sp',
            text_size=(None, None),
            halign='center',
            valign='middle'
        )
        layout.add_widget(self.translation_label)
        
        # Control buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=60, spacing=10)
        
        self.record_button = Button(
            text="Start Listening",
            size_hint_x=0.5,
            background_color=(0.2, 0.8, 0.2, 1)
        )
        self.record_button.bind(on_press=self.toggle_recording)
        button_layout.add_widget(self.record_button)
        
        settings_button = Button(
            text="Settings",
            size_hint_x=0.5,
            background_color=(0.2, 0.6, 0.8, 1)
        )
        settings_button.bind(on_press=self.go_to_settings)
        button_layout.add_widget(settings_button)
        
        layout.add_widget(button_layout)
        
        # Navigation buttons
        nav_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, spacing=10)
        
        training_button = Button(
            text="Training Mode",
            size_hint_x=0.5,
            background_color=(0.8, 0.6, 0.2, 1)
        )
        training_button.bind(on_press=self.go_to_training)
        nav_layout.add_widget(training_button)
        
        viewer_button = Button(
            text="Model Viewer",
            size_hint_x=0.5,
            background_color=(0.6, 0.2, 0.8, 1)
        )
        viewer_button.bind(on_press=self.go_to_viewer)
        nav_layout.add_widget(viewer_button)
        
        layout.add_widget(nav_layout)
        
        self.add_widget(layout)
        
        # Bind properties to UI updates
        self.bind(status_text=self._update_status)
        self.bind(confidence=self._update_confidence)
        self.bind(is_recording=self._update_recording_state)
        self.bind(current_translation=self._update_translation)
    
    def toggle_recording(self, instance):
        """Toggle recording on/off"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording and processing audio"""
        if self.predictor.model is None:
            self.status_text = "Please train a model first"
            return
        
        self.should_record = True
        self.is_recording = True
        self.record_button.text = "Stop Listening"
        self.record_button.background_color = (0.8, 0.2, 0.2, 1)
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.status_text = "Listening for vocal gestures..."
    
    def stop_recording(self):
        """Stop recording"""
        self.should_record = False
        self.is_recording = False
        self.record_button.text = "Start Listening"
        self.record_button.background_color = (0.2, 0.8, 0.2, 1)
        self.status_text = "Recording stopped"
        
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
    
    def _recording_loop(self):
        """Main recording and processing loop"""
        try:
            with self.recorder.start_stream():
                while self.should_record:
                    # Capture audio chunk
                    audio_data = self.recorder.get_audio_chunk()
                    
                    if audio_data is not None and len(audio_data) > 0:
                        # Process audio
                        features = self.processor.extract_features(audio_data)
                        
                        if features is not None:
                            # Predict sound
                            prediction, confidence = self.predictor.predict(features)
                            
                            # Update UI on main thread
                            Clock.schedule_once(
                                lambda dt: self._update_prediction(prediction, confidence)
                            )
                    
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                    
        except Exception as e:
            Clock.schedule_once(
                lambda dt: self._handle_recording_error(str(e))
            )
    
    def _update_prediction(self, prediction, confidence):
        """Update prediction display"""
        self.confidence = confidence
        self.current_translation = prediction
        
        # Update confidence bar
        self.confidence_bar.value = confidence
    
    def _handle_recording_error(self, error_msg):
        """Handle recording errors"""
        self.status_text = f"Recording error: {error_msg}"
        self.stop_recording()
    
    def _update_status(self, instance, value):
        """Update status label"""
        self.status_label.text = value
    
    def _update_confidence(self, instance, value):
        """Update confidence bar"""
        self.confidence_bar.value = value
    
    def _update_recording_state(self, instance, value):
        """Update recording state"""
        pass  # Already handled in toggle_recording
    
    def _update_translation(self, instance, value):
        """Update translation display"""
        self.translation_label.text = value
    
    def go_to_training(self, instance):
        """Navigate to training screen"""
        self.manager.current = 'training'
    
    def go_to_viewer(self, instance):
        """Navigate to model viewer screen"""
        self.manager.current = 'model_viewer'
    
    def go_to_settings(self, instance):
        """Navigate to settings screen"""
        self.manager.current = 'settings'
    
    def on_enter(self):
        """Called when screen becomes active"""
        # Refresh model status
        model_path = os.path.join('data', 'models', 'user_model.pth')
        if os.path.exists(model_path):
            self.status_text = "Model loaded and ready"
        else:
            self.status_text = "No trained model found. Please train first."
    
    def on_leave(self):
        """Called when leaving the screen"""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording() 