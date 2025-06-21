"""
Training screen for Sonit application
Allows users to record and label vocal gestures
"""

import os
import threading
import time
from datetime import datetime
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.properties import StringProperty, BooleanProperty, NumericProperty
from kivy.core.window import Window

# Import audio and model components
from audio.recorder import AudioRecorder
from audio.processor import AudioProcessor
from model.trainer import ModelTrainer
from utils.database import DatabaseManager


class TrainingScreen(Screen):
    """Training screen for recording and labeling vocal gestures"""
    
    # Properties for UI updates
    status_text = StringProperty("Ready to record samples...")
    is_recording = BooleanProperty(False)
    sample_count = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recorder = None
        self.processor = None
        self.trainer = None
        self.db_manager = None
        self.recording_thread = None
        self.should_record = False
        self.current_audio_data = None
        self.samples = []  # List of (features, label) tuples
        
        # Initialize components
        self._init_components()
        self._setup_ui()
        self._load_existing_samples()
    
    def _init_components(self):
        """Initialize audio and model components"""
        try:
            self.recorder = AudioRecorder()
            self.processor = AudioProcessor()
            self.trainer = ModelTrainer()
            self.db_manager = DatabaseManager()
            
            # Create data directories if they don't exist
            os.makedirs('data/recordings', exist_ok=True)
            os.makedirs('data/models', exist_ok=True)
            
        except Exception as e:
            self.status_text = f"Error initializing: {str(e)}"
    
    def _setup_ui(self):
        """Setup the user interface"""
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        # Title
        title = Label(
            text="Training Mode - Record Vocal Gestures",
            size_hint_y=None,
            height=50,
            font_size='20sp',
            bold=True
        )
        layout.add_widget(title)
        
        # Status display
        self.status_label = Label(
            text=self.status_text,
            size_hint_y=None,
            height=40,
            font_size='14sp'
        )
        layout.add_widget(self.status_label)
        
        # Sample counter
        self.sample_label = Label(
            text=f"Recorded samples: {self.sample_count}",
            size_hint_y=None,
            height=30,
            font_size='14sp'
        )
        layout.add_widget(self.sample_label)
        
        # Recording controls
        record_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=60, spacing=10)
        
        self.record_button = Button(
            text="Record Sample",
            size_hint_x=0.5,
            background_color=(0.2, 0.8, 0.2, 1)
        )
        self.record_button.bind(on_press=self.toggle_recording)
        record_layout.add_widget(self.record_button)
        
        clear_button = Button(
            text="Clear All",
            size_hint_x=0.5,
            background_color=(0.8, 0.2, 0.2, 1)
        )
        clear_button.bind(on_press=self.clear_samples)
        record_layout.add_widget(clear_button)
        
        layout.add_widget(record_layout)
        
        # Label input section
        label_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=120, spacing=5)
        
        label_layout.add_widget(Label(
            text="Label this sound:",
            size_hint_y=None,
            height=25,
            font_size='14sp'
        ))
        
        # Quick label buttons
        quick_labels = ["Yes", "No", "Maybe", "Help", "Stop", "Continue", "Good", "Bad"]
        quick_layout = GridLayout(cols=4, size_hint_y=None, height=40, spacing=5)
        
        for label in quick_labels:
            btn = Button(
                text=label,
                size_hint_y=None,
                height=35,
                background_color=(0.6, 0.6, 0.6, 1)
            )
            btn.bind(on_press=lambda btn, l=label: self.set_label(l))
            quick_layout.add_widget(btn)
        
        label_layout.add_widget(quick_layout)
        
        # Custom label input
        input_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40, spacing=10)
        input_layout.add_widget(Label(text="Custom:", size_hint_x=0.2))
        
        self.label_input = TextInput(
            multiline=False,
            size_hint_x=0.6,
            hint_text="Enter custom label..."
        )
        input_layout.add_widget(self.label_input)
        
        save_button = Button(
            text="Save",
            size_hint_x=0.2,
            background_color=(0.2, 0.6, 0.8, 1)
        )
        save_button.bind(on_press=self.save_sample)
        input_layout.add_widget(save_button)
        
        label_layout.add_widget(input_layout)
        layout.add_widget(label_layout)
        
        # Training controls
        train_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=60, spacing=10)
        
        train_button = Button(
            text="Train Model",
            size_hint_x=0.5,
            background_color=(0.8, 0.6, 0.2, 1)
        )
        train_button.bind(on_press=self.train_model)
        train_layout.add_widget(train_button)
        
        back_button = Button(
            text="Back to Main",
            size_hint_x=0.5,
            background_color=(0.6, 0.6, 0.6, 1)
        )
        back_button.bind(on_press=self.go_back)
        train_layout.add_widget(back_button)
        
        layout.add_widget(train_layout)
        
        # Sample list
        list_label = Label(
            text="Recorded Samples:",
            size_hint_y=None,
            height=30,
            font_size='14sp',
            bold=True
        )
        layout.add_widget(list_label)
        
        # Scrollable sample list
        scroll = ScrollView(size_hint=(1, 1))
        self.sample_grid = GridLayout(cols=3, spacing=5, size_hint_y=None)
        self.sample_grid.bind(minimum_height=self.sample_grid.setter('height'))
        scroll.add_widget(self.sample_grid)
        layout.add_widget(scroll)
        
        self.add_widget(layout)
        
        # Bind properties to UI updates
        self.bind(status_text=self._update_status)
        self.bind(sample_count=self._update_sample_count)
        self.bind(is_recording=self._update_recording_state)
    
    def toggle_recording(self, instance):
        """Toggle recording on/off"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording a sample"""
        self.should_record = True
        self.is_recording = True
        self.record_button.text = "Stop Recording"
        self.record_button.background_color = (0.8, 0.2, 0.2, 1)
        self.status_text = "Recording... Make your sound now!"
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record_sample)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop recording"""
        self.should_record = False
        self.is_recording = False
        self.record_button.text = "Record Sample"
        self.record_button.background_color = (0.2, 0.8, 0.2, 1)
        self.status_text = "Recording stopped. Add a label and save."
        
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
    
    def _record_sample(self):
        """Record a single sample"""
        try:
            with self.recorder.start_stream():
                # Record for 2 seconds
                start_time = time.time()
                audio_chunks = []
                
                while self.should_record and (time.time() - start_time) < 2.0:
                    chunk = self.recorder.get_audio_chunk()
                    if chunk is not None:
                        audio_chunks.append(chunk)
                    time.sleep(0.01)
                
                if audio_chunks:
                    # Combine chunks and process
                    combined_audio = b''.join(audio_chunks)
                    self.current_audio_data = combined_audio
                    
                    # Extract features
                    features = self.processor.extract_features(combined_audio)
                    if features is not None:
                        self.current_features = features
                        Clock.schedule_once(lambda dt: self._recording_complete())
                    else:
                        Clock.schedule_once(lambda dt: self._recording_error("No audio features detected"))
                else:
                    Clock.schedule_once(lambda dt: self._recording_error("No audio recorded"))
                    
        except Exception as e:
            Clock.schedule_once(lambda dt: self._recording_error(str(e)))
    
    def _recording_complete(self):
        """Called when recording is complete"""
        self.status_text = "Sample recorded! Add a label and save."
    
    def _recording_error(self, error_msg):
        """Handle recording errors"""
        self.status_text = f"Recording error: {error_msg}"
        self.stop_recording()
    
    def set_label(self, label):
        """Set the label for the current sample"""
        self.label_input.text = label
    
    def save_sample(self, instance):
        """Save the current sample with its label"""
        if not hasattr(self, 'current_features'):
            self.status_text = "No sample to save. Record first."
            return
        
        label = self.label_input.text.strip()
        if not label:
            self.status_text = "Please enter a label for the sample."
            return
        
        try:
            # Save to database
            sample_id = self.db_manager.save_sample(self.current_features, label)
            
            # Add to local list
            self.samples.append((self.current_features, label))
            self.sample_count = len(self.samples)
            
            # Add to UI list
            self._add_sample_to_ui(label, sample_id)
            
            # Clear current sample
            self.current_features = None
            self.current_audio_data = None
            self.label_input.text = ""
            
            self.status_text = f"Sample saved as '{label}'"
            
        except Exception as e:
            self.status_text = f"Error saving sample: {str(e)}"
    
    def _add_sample_to_ui(self, label, sample_id):
        """Add a sample to the UI list"""
        # Create sample display
        sample_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=60, spacing=2)
        
        # Label
        label_widget = Label(
            text=label,
            size_hint_y=None,
            height=25,
            font_size='12sp',
            bold=True
        )
        sample_layout.add_widget(label_widget)
        
        # Sample ID and delete button
        info_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=25, spacing=5)
        
        info_layout.add_widget(Label(
            text=f"ID: {sample_id}",
            size_hint_x=0.7,
            font_size='10sp'
        ))
        
        delete_btn = Button(
            text="X",
            size_hint_x=0.3,
            size_hint_y=None,
            height=20,
            background_color=(0.8, 0.2, 0.2, 1),
            font_size='10sp'
        )
        delete_btn.bind(on_press=lambda btn, sid=sample_id: self.delete_sample(sid))
        info_layout.add_widget(delete_btn)
        
        sample_layout.add_widget(info_layout)
        self.sample_grid.add_widget(sample_layout)
    
    def delete_sample(self, sample_id):
        """Delete a sample"""
        try:
            self.db_manager.delete_sample(sample_id)
            self._load_existing_samples()  # Refresh the list
            self.status_text = f"Sample {sample_id} deleted"
        except Exception as e:
            self.status_text = f"Error deleting sample: {str(e)}"
    
    def clear_samples(self, instance):
        """Clear all samples"""
        try:
            self.db_manager.clear_all_samples()
            self.samples = []
            self.sample_count = 0
            self.sample_grid.clear_widgets()
            self.status_text = "All samples cleared"
        except Exception as e:
            self.status_text = f"Error clearing samples: {str(e)}"
    
    def train_model(self, instance):
        """Train the model with recorded samples"""
        if self.sample_count < 5:
            self.status_text = "Need at least 5 samples to train. Record more samples."
            return
        
        try:
            self.status_text = "Training model... This may take a while."
            
            # Get all samples from database
            samples = self.db_manager.get_all_samples()
            
            if len(samples) < 5:
                self.status_text = "Need at least 5 samples to train."
                return
            
            # Train model in separate thread
            training_thread = threading.Thread(
                target=self._train_model_thread,
                args=(samples,)
            )
            training_thread.daemon = True
            training_thread.start()
            
        except Exception as e:
            self.status_text = f"Error starting training: {str(e)}"
    
    def _train_model_thread(self, samples):
        """Train model in background thread"""
        try:
            # Prepare training data
            features = [sample[0] for sample in samples]
            labels = [sample[1] for sample in samples]
            
            # Train model
            model_path = os.path.join('data', 'models', 'user_model.pth')
            self.trainer.train(features, labels, model_path)
            
            # Update UI on main thread
            Clock.schedule_once(lambda dt: self._training_complete())
            
        except Exception as e:
            Clock.schedule_once(lambda dt: self._training_error(str(e)))
    
    def _training_complete(self):
        """Called when training is complete"""
        self.status_text = "Model trained successfully! You can now use it for translation."
    
    def _training_error(self, error_msg):
        """Handle training errors"""
        self.status_text = f"Training error: {error_msg}"
    
    def _load_existing_samples(self):
        """Load existing samples from database"""
        try:
            samples = self.db_manager.get_all_samples()
            self.samples = samples
            self.sample_count = len(samples)
            
            # Clear and rebuild UI list
            self.sample_grid.clear_widgets()
            for sample_id, features, label in samples:
                self._add_sample_to_ui(label, sample_id)
                
        except Exception as e:
            self.status_text = f"Error loading samples: {str(e)}"
    
    def _update_status(self, instance, value):
        """Update status label"""
        self.status_label.text = value
    
    def _update_sample_count(self, instance, value):
        """Update sample count label"""
        self.sample_label.text = f"Recorded samples: {value}"
    
    def _update_recording_state(self, instance, value):
        """Update recording state"""
        pass  # Already handled in toggle_recording
    
    def go_back(self, instance):
        """Go back to main screen"""
        self.manager.current = 'main'
    
    def on_enter(self):
        """Called when screen becomes active"""
        self._load_existing_samples()
    
    def on_leave(self):
        """Called when leaving the screen"""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording() 