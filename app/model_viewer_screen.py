"""
Model viewer screen for Sonit application
Displays sound embeddings, confidence levels, and live output
"""

import os
import threading
import time
import numpy as np
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.clock import Clock
from kivy.properties import StringProperty, BooleanProperty, NumericProperty
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure

# Import audio and model components
from audio.recorder import AudioRecorder
from audio.processor import AudioProcessor
from model.predictor import SoundPredictor
from utils.database import DatabaseManager


class ModelViewerScreen(Screen):
    """Model viewer screen for visualizing sound analysis"""
    
    # Properties for UI updates
    status_text = StringProperty("Model viewer ready")
    is_live = BooleanProperty(False)
    current_prediction = StringProperty("")
    current_confidence = NumericProperty(0.0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recorder = None
        self.processor = None
        self.predictor = None
        self.db_manager = None
        self.live_thread = None
        self.should_live = False
        
        # Visualization components
        self.fig = None
        self.canvas_widget = None
        self.spectrogram_ax = None
        self.confidence_ax = None
        self.prediction_ax = None
        
        # Data for visualization
        self.audio_history = []
        self.confidence_history = []
        self.prediction_history = []
        
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
            
            # Load model if available
            model_path = os.path.join('data', 'models', 'user_model.pth')
            if os.path.exists(model_path):
                self.predictor.load_model(model_path)
                self.status_text = "Model loaded for visualization"
            else:
                self.status_text = "No trained model found"
                
        except Exception as e:
            self.status_text = f"Error initializing: {str(e)}"
    
    def _setup_ui(self):
        """Setup the user interface"""
        layout = BoxLayout(orientation='vertical', padding=10, spacing=5)
        
        # Title
        title = Label(
            text="Model Viewer - Sound Analysis",
            size_hint_y=None,
            height=40,
            font_size='18sp',
            bold=True
        )
        layout.add_widget(title)
        
        # Status display
        self.status_label = Label(
            text=self.status_text,
            size_hint_y=None,
            height=30,
            font_size='12sp'
        )
        layout.add_widget(self.status_label)
        
        # Control buttons
        control_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, spacing=10)
        
        self.live_button = ToggleButton(
            text="Start Live View",
            size_hint_x=0.5,
            background_color=(0.2, 0.8, 0.2, 1)
        )
        self.live_button.bind(on_press=self.toggle_live_view)
        control_layout.add_widget(self.live_button)
        
        back_button = Button(
            text="Back to Main",
            size_hint_x=0.5,
            background_color=(0.6, 0.6, 0.6, 1)
        )
        back_button.bind(on_press=self.go_back)
        control_layout.add_widget(back_button)
        
        layout.add_widget(control_layout)
        
        # Current prediction display
        pred_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=80, spacing=5)
        
        pred_layout.add_widget(Label(
            text="Current Prediction:",
            size_hint_y=None,
            height=25,
            font_size='14sp',
            bold=True
        ))
        
        self.prediction_label = Label(
            text="No prediction yet",
            size_hint_y=None,
            height=25,
            font_size='16sp'
        )
        pred_layout.add_widget(self.prediction_label)
        
        # Confidence bar
        conf_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=25, spacing=5)
        conf_layout.add_widget(Label(text="Confidence:", size_hint_x=0.3, font_size='12sp'))
        self.confidence_bar = ProgressBar(max=1.0, size_hint_x=0.7)
        conf_layout.add_widget(self.confidence_bar)
        pred_layout.add_widget(conf_layout)
        
        layout.add_widget(pred_layout)
        
        # Visualization area
        viz_label = Label(
            text="Sound Analysis Visualization",
            size_hint_y=None,
            height=25,
            font_size='14sp',
            bold=True
        )
        layout.add_widget(viz_label)
        
        # Create matplotlib figure
        self._create_visualization()
        
        # Add the matplotlib canvas
        if self.canvas_widget:
            layout.add_widget(self.canvas_widget)
        
        self.add_widget(layout)
        
        # Bind properties to UI updates
        self.bind(status_text=self._update_status)
        self.bind(current_prediction=self._update_prediction)
        self.bind(current_confidence=self._update_confidence)
        self.bind(is_live=self._update_live_state)
    
    def _create_visualization(self):
        """Create matplotlib visualization"""
        try:
            # Create figure with subplots
            self.fig = Figure(figsize=(8, 6), dpi=100)
            
            # Spectrogram subplot
            self.spectrogram_ax = self.fig.add_subplot(3, 1, 1)
            self.spectrogram_ax.set_title('Audio Spectrogram')
            self.spectrogram_ax.set_ylabel('Frequency (Hz)')
            
            # Confidence history subplot
            self.confidence_ax = self.fig.add_subplot(3, 1, 2)
            self.confidence_ax.set_title('Confidence Over Time')
            self.confidence_ax.set_ylabel('Confidence')
            self.confidence_ax.set_ylim(0, 1)
            
            # Prediction history subplot
            self.prediction_ax = self.fig.add_subplot(3, 1, 3)
            self.prediction_ax.set_title('Prediction History')
            self.prediction_ax.set_ylabel('Prediction')
            
            # Adjust layout
            self.fig.tight_layout()
            
            # Create canvas widget
            self.canvas_widget = FigureCanvasKivyAgg(self.fig)
            
        except Exception as e:
            self.status_text = f"Error creating visualization: {str(e)}"
    
    def toggle_live_view(self, instance):
        """Toggle live visualization on/off"""
        if not self.is_live:
            self.start_live_view()
        else:
            self.stop_live_view()
    
    def start_live_view(self):
        """Start live visualization"""
        if self.predictor.model is None:
            self.status_text = "No model loaded for visualization"
            return
        
        self.should_live = True
        self.is_live = True
        self.live_button.text = "Stop Live View"
        self.live_button.background_color = (0.8, 0.2, 0.2, 1)
        
        # Start live processing in separate thread
        self.live_thread = threading.Thread(target=self._live_processing_loop)
        self.live_thread.daemon = True
        self.live_thread.start()
        
        self.status_text = "Live visualization active..."
    
    def stop_live_view(self):
        """Stop live visualization"""
        self.should_live = False
        self.is_live = False
        self.live_button.text = "Start Live View"
        self.live_button.background_color = (0.2, 0.8, 0.2, 1)
        self.status_text = "Live visualization stopped"
        
        if self.live_thread:
            self.live_thread.join(timeout=1.0)
    
    def _live_processing_loop(self):
        """Main live processing loop"""
        try:
            with self.recorder.start_stream():
                while self.should_live:
                    # Capture audio chunk
                    audio_data = self.recorder.get_audio_chunk()
                    
                    if audio_data is not None and len(audio_data) > 0:
                        # Process audio
                        features = self.processor.extract_features(audio_data)
                        
                        if features is not None:
                            # Predict sound
                            prediction, confidence = self.predictor.predict(features)
                            
                            # Store data for visualization
                            self.audio_history.append(audio_data)
                            self.confidence_history.append(confidence)
                            self.prediction_history.append(prediction)
                            
                            # Keep only last 100 samples
                            if len(self.audio_history) > 100:
                                self.audio_history.pop(0)
                                self.confidence_history.pop(0)
                                self.prediction_history.pop(0)
                            
                            # Update UI on main thread
                            Clock.schedule_once(
                                lambda dt: self._update_live_data(prediction, confidence, audio_data)
                            )
                            
                            # Update visualization
                            Clock.schedule_once(lambda dt: self._update_visualization())
                    
                    time.sleep(0.1)  # Small delay
                    
        except Exception as e:
            Clock.schedule_once(
                lambda dt: self._handle_live_error(str(e))
            )
    
    def _update_live_data(self, prediction, confidence, audio_data):
        """Update live data display"""
        self.current_prediction = prediction
        self.current_confidence = confidence
    
    def _update_visualization(self):
        """Update the visualization plots"""
        try:
            if not self.fig or len(self.audio_history) == 0:
                return
            
            # Clear previous plots
            self.spectrogram_ax.clear()
            self.confidence_ax.clear()
            self.prediction_ax.clear()
            
            # Update spectrogram (show last audio sample)
            if len(self.audio_history) > 0:
                last_audio = self.audio_history[-1]
                # Convert bytes to numpy array (assuming 16-bit PCM)
                audio_array = np.frombuffer(last_audio, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Create spectrogram
                self.spectrogram_ax.specgram(audio_array, Fs=16000, cmap='viridis')
                self.spectrogram_ax.set_title('Audio Spectrogram')
                self.spectrogram_ax.set_ylabel('Frequency (Hz)')
            
            # Update confidence history
            if len(self.confidence_history) > 0:
                time_points = range(len(self.confidence_history))
                self.confidence_ax.plot(time_points, self.confidence_history, 'b-', linewidth=2)
                self.confidence_ax.set_title('Confidence Over Time')
                self.confidence_ax.set_ylabel('Confidence')
                self.confidence_ax.set_ylim(0, 1)
                self.confidence_ax.grid(True, alpha=0.3)
            
            # Update prediction history
            if len(self.prediction_history) > 0:
                time_points = range(len(self.prediction_history))
                # Convert predictions to numeric for plotting
                unique_predictions = list(set(self.prediction_history))
                pred_numeric = [unique_predictions.index(p) for p in self.prediction_history]
                
                self.prediction_ax.plot(time_points, pred_numeric, 'g-', linewidth=2)
                self.prediction_ax.set_title('Prediction History')
                self.prediction_ax.set_ylabel('Prediction')
                self.prediction_ax.set_yticks(range(len(unique_predictions)))
                self.prediction_ax.set_yticklabels(unique_predictions)
                self.prediction_ax.grid(True, alpha=0.3)
            
            # Redraw canvas
            if self.canvas_widget:
                self.canvas_widget.draw()
                
        except Exception as e:
            self.status_text = f"Visualization error: {str(e)}"
    
    def _handle_live_error(self, error_msg):
        """Handle live processing errors"""
        self.status_text = f"Live processing error: {error_msg}"
        self.stop_live_view()
    
    def _update_status(self, instance, value):
        """Update status label"""
        self.status_label.text = value
    
    def _update_prediction(self, instance, value):
        """Update prediction label"""
        self.prediction_label.text = value
    
    def _update_confidence(self, instance, value):
        """Update confidence bar"""
        self.confidence_bar.value = value
    
    def _update_live_state(self, instance, value):
        """Update live state"""
        pass  # Already handled in toggle_live_view
    
    def go_back(self, instance):
        """Go back to main screen"""
        self.manager.current = 'main'
    
    def on_enter(self):
        """Called when screen becomes active"""
        # Check if model is available
        model_path = os.path.join('data', 'models', 'user_model.pth')
        if os.path.exists(model_path):
            self.status_text = "Model loaded for visualization"
        else:
            self.status_text = "No trained model found"
    
    def on_leave(self):
        """Called when leaving the screen"""
        # Stop live view if active
        if self.is_live:
            self.stop_live_view() 