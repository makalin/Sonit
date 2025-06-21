"""
Settings screen for Sonit application
Configure audio parameters, model settings, and preferences
"""

import os
import json
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.slider import Slider
from kivy.uix.checkbox import CheckBox
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty, BooleanProperty, NumericProperty


class SettingsScreen(Screen):
    """Settings screen for application configuration"""
    
    # Properties for settings
    sample_rate = NumericProperty(16000)
    chunk_size = NumericProperty(1024)
    sensitivity = NumericProperty(0.5)
    auto_save = BooleanProperty(True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings_file = 'data/settings.json'
        self.settings = {}
        
        # Initialize settings
        self._load_settings()
        self._setup_ui()
    
    def _load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    self.settings = json.load(f)
            else:
                # Default settings
                self.settings = {
                    'audio': {
                        'sample_rate': 16000,
                        'chunk_size': 1024,
                        'channels': 1,
                        'format': 'int16'
                    },
                    'model': {
                        'confidence_threshold': 0.5,
                        'max_predictions': 5
                    },
                    'ui': {
                        'auto_save': True,
                        'show_confidence': True,
                        'dark_mode': False
                    },
                    'training': {
                        'min_samples': 5,
                        'max_samples': 1000,
                        'validation_split': 0.2
                    }
                }
                self._save_settings()
                
        except Exception as e:
            print(f"Error loading settings: {e}")
            # Use default settings
            self.settings = {}
    
    def _save_settings(self):
        """Save settings to file"""
        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def _setup_ui(self):
        """Setup the user interface"""
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        # Title
        title = Label(
            text="Settings",
            size_hint_y=None,
            height=50,
            font_size='24sp',
            bold=True
        )
        layout.add_widget(title)
        
        # Scrollable content
        scroll = ScrollView(size_hint=(1, 1))
        content = BoxLayout(orientation='vertical', spacing=15, size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))
        
        # Audio Settings Section
        content.add_widget(self._create_section_title("Audio Settings"))
        content.add_widget(self._create_audio_settings())
        
        # Model Settings Section
        content.add_widget(self._create_section_title("Model Settings"))
        content.add_widget(self._create_model_settings())
        
        # UI Settings Section
        content.add_widget(self._create_section_title("Interface Settings"))
        content.add_widget(self._create_ui_settings())
        
        # Training Settings Section
        content.add_widget(self._create_section_title("Training Settings"))
        content.add_widget(self._create_training_settings())
        
        # Control buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=60, spacing=10)
        
        save_button = Button(
            text="Save Settings",
            size_hint_x=0.5,
            background_color=(0.2, 0.8, 0.2, 1)
        )
        save_button.bind(on_press=self.save_settings)
        button_layout.add_widget(save_button)
        
        back_button = Button(
            text="Back to Main",
            size_hint_x=0.5,
            background_color=(0.6, 0.6, 0.6, 1)
        )
        back_button.bind(on_press=self.go_back)
        button_layout.add_widget(back_button)
        
        content.add_widget(button_layout)
        
        scroll.add_widget(content)
        layout.add_widget(scroll)
        
        self.add_widget(layout)
    
    def _create_section_title(self, title):
        """Create a section title"""
        return Label(
            text=title,
            size_hint_y=None,
            height=30,
            font_size='16sp',
            bold=True,
            color=(0.2, 0.6, 0.8, 1)
        )
    
    def _create_audio_settings(self):
        """Create audio settings section"""
        layout = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None, height=200)
        
        # Sample Rate
        sample_rate_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        sample_rate_layout.add_widget(Label(text="Sample Rate:", size_hint_x=0.4))
        
        sample_rate_spinner = Spinner(
            text=str(self.settings.get('audio', {}).get('sample_rate', 16000)),
            values=['8000', '16000', '22050', '44100'],
            size_hint_x=0.6
        )
        sample_rate_spinner.bind(text=self._on_sample_rate_change)
        sample_rate_layout.add_widget(sample_rate_spinner)
        layout.add_widget(sample_rate_layout)
        
        # Chunk Size
        chunk_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        chunk_layout.add_widget(Label(text="Chunk Size:", size_hint_x=0.4))
        
        chunk_spinner = Spinner(
            text=str(self.settings.get('audio', {}).get('chunk_size', 1024)),
            values=['512', '1024', '2048', '4096'],
            size_hint_x=0.6
        )
        chunk_spinner.bind(text=self._on_chunk_size_change)
        chunk_layout.add_widget(chunk_spinner)
        layout.add_widget(chunk_layout)
        
        # Channels
        channels_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        channels_layout.add_widget(Label(text="Channels:", size_hint_x=0.4))
        
        channels_spinner = Spinner(
            text=str(self.settings.get('audio', {}).get('channels', 1)),
            values=['1', '2'],
            size_hint_x=0.6
        )
        channels_spinner.bind(text=self._on_channels_change)
        channels_layout.add_widget(channels_spinner)
        layout.add_widget(channels_layout)
        
        # Format
        format_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        format_layout.add_widget(Label(text="Format:", size_hint_x=0.4))
        
        format_spinner = Spinner(
            text=self.settings.get('audio', {}).get('format', 'int16'),
            values=['int16', 'int32', 'float32'],
            size_hint_x=0.6
        )
        format_spinner.bind(text=self._on_format_change)
        format_layout.add_widget(format_spinner)
        layout.add_widget(format_layout)
        
        return layout
    
    def _create_model_settings(self):
        """Create model settings section"""
        layout = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None, height=120)
        
        # Confidence Threshold
        conf_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=60)
        conf_layout.add_widget(Label(text="Confidence Threshold:", size_hint_y=None, height=25))
        
        conf_slider = Slider(
            min=0.1,
            max=0.9,
            value=self.settings.get('model', {}).get('confidence_threshold', 0.5),
            size_hint_y=None,
            height=30
        )
        conf_slider.bind(value=self._on_confidence_threshold_change)
        conf_layout.add_widget(conf_slider)
        layout.add_widget(conf_layout)
        
        # Max Predictions
        max_pred_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        max_pred_layout.add_widget(Label(text="Max Predictions:", size_hint_x=0.4))
        
        max_pred_spinner = Spinner(
            text=str(self.settings.get('model', {}).get('max_predictions', 5)),
            values=['1', '3', '5', '10'],
            size_hint_x=0.6
        )
        max_pred_spinner.bind(text=self._on_max_predictions_change)
        max_pred_layout.add_widget(max_pred_spinner)
        layout.add_widget(max_pred_layout)
        
        return layout
    
    def _create_ui_settings(self):
        """Create UI settings section"""
        layout = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None, height=150)
        
        # Auto Save
        auto_save_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        auto_save_layout.add_widget(Label(text="Auto Save:", size_hint_x=0.7))
        
        auto_save_checkbox = CheckBox(
            active=self.settings.get('ui', {}).get('auto_save', True),
            size_hint_x=0.3
        )
        auto_save_checkbox.bind(active=self._on_auto_save_change)
        auto_save_layout.add_widget(auto_save_checkbox)
        layout.add_widget(auto_save_layout)
        
        # Show Confidence
        show_conf_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        show_conf_layout.add_widget(Label(text="Show Confidence:", size_hint_x=0.7))
        
        show_conf_checkbox = CheckBox(
            active=self.settings.get('ui', {}).get('show_confidence', True),
            size_hint_x=0.3
        )
        show_conf_checkbox.bind(active=self._on_show_confidence_change)
        show_conf_layout.add_widget(show_conf_checkbox)
        layout.add_widget(show_conf_layout)
        
        # Dark Mode
        dark_mode_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        dark_mode_layout.add_widget(Label(text="Dark Mode:", size_hint_x=0.7))
        
        dark_mode_checkbox = CheckBox(
            active=self.settings.get('ui', {}).get('dark_mode', False),
            size_hint_x=0.3
        )
        dark_mode_checkbox.bind(active=self._on_dark_mode_change)
        dark_mode_layout.add_widget(dark_mode_checkbox)
        layout.add_widget(dark_mode_layout)
        
        return layout
    
    def _create_training_settings(self):
        """Create training settings section"""
        layout = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None, height=150)
        
        # Min Samples
        min_samples_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        min_samples_layout.add_widget(Label(text="Min Samples:", size_hint_x=0.4))
        
        min_samples_spinner = Spinner(
            text=str(self.settings.get('training', {}).get('min_samples', 5)),
            values=['3', '5', '10', '20'],
            size_hint_x=0.6
        )
        min_samples_spinner.bind(text=self._on_min_samples_change)
        min_samples_layout.add_widget(min_samples_spinner)
        layout.add_widget(min_samples_layout)
        
        # Max Samples
        max_samples_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        max_samples_layout.add_widget(Label(text="Max Samples:", size_hint_x=0.4))
        
        max_samples_spinner = Spinner(
            text=str(self.settings.get('training', {}).get('max_samples', 1000)),
            values=['100', '500', '1000', '2000'],
            size_hint_x=0.6
        )
        max_samples_spinner.bind(text=self._on_max_samples_change)
        max_samples_layout.add_widget(max_samples_spinner)
        layout.add_widget(max_samples_layout)
        
        # Validation Split
        val_split_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        val_split_layout.add_widget(Label(text="Validation Split:", size_hint_x=0.4))
        
        val_split_spinner = Spinner(
            text=str(self.settings.get('training', {}).get('validation_split', 0.2)),
            values=['0.1', '0.2', '0.3', '0.4'],
            size_hint_x=0.6
        )
        val_split_spinner.bind(text=self._on_validation_split_change)
        val_split_layout.add_widget(val_split_spinner)
        layout.add_widget(val_split_layout)
        
        return layout
    
    # Settings change handlers
    def _on_sample_rate_change(self, spinner, value):
        """Handle sample rate change"""
        if 'audio' not in self.settings:
            self.settings['audio'] = {}
        self.settings['audio']['sample_rate'] = int(value)
    
    def _on_chunk_size_change(self, spinner, value):
        """Handle chunk size change"""
        if 'audio' not in self.settings:
            self.settings['audio'] = {}
        self.settings['audio']['chunk_size'] = int(value)
    
    def _on_channels_change(self, spinner, value):
        """Handle channels change"""
        if 'audio' not in self.settings:
            self.settings['audio'] = {}
        self.settings['audio']['channels'] = int(value)
    
    def _on_format_change(self, spinner, value):
        """Handle format change"""
        if 'audio' not in self.settings:
            self.settings['audio'] = {}
        self.settings['audio']['format'] = value
    
    def _on_confidence_threshold_change(self, slider, value):
        """Handle confidence threshold change"""
        if 'model' not in self.settings:
            self.settings['model'] = {}
        self.settings['model']['confidence_threshold'] = value
    
    def _on_max_predictions_change(self, spinner, value):
        """Handle max predictions change"""
        if 'model' not in self.settings:
            self.settings['model'] = {}
        self.settings['model']['max_predictions'] = int(value)
    
    def _on_auto_save_change(self, checkbox, value):
        """Handle auto save change"""
        if 'ui' not in self.settings:
            self.settings['ui'] = {}
        self.settings['ui']['auto_save'] = value
    
    def _on_show_confidence_change(self, checkbox, value):
        """Handle show confidence change"""
        if 'ui' not in self.settings:
            self.settings['ui'] = {}
        self.settings['ui']['show_confidence'] = value
    
    def _on_dark_mode_change(self, checkbox, value):
        """Handle dark mode change"""
        if 'ui' not in self.settings:
            self.settings['ui'] = {}
        self.settings['ui']['dark_mode'] = value
    
    def _on_min_samples_change(self, spinner, value):
        """Handle min samples change"""
        if 'training' not in self.settings:
            self.settings['training'] = {}
        self.settings['training']['min_samples'] = int(value)
    
    def _on_max_samples_change(self, spinner, value):
        """Handle max samples change"""
        if 'training' not in self.settings:
            self.settings['training'] = {}
        self.settings['training']['max_samples'] = int(value)
    
    def _on_validation_split_change(self, spinner, value):
        """Handle validation split change"""
        if 'training' not in self.settings:
            self.settings['training'] = {}
        self.settings['training']['validation_split'] = float(value)
    
    def save_settings(self, instance):
        """Save current settings"""
        try:
            self._save_settings()
            # Update status (you could add a status label to show this)
            print("Settings saved successfully!")
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def go_back(self, instance):
        """Go back to main screen"""
        self.manager.current = 'main'
    
    def get_setting(self, category, key, default=None):
        """Get a setting value"""
        return self.settings.get(category, {}).get(key, default)
    
    def set_setting(self, category, key, value):
        """Set a setting value"""
        if category not in self.settings:
            self.settings[category] = {}
        self.settings[category][key] = value 