#!/usr/bin/env python3
"""
Sonit - Translating the Unspoken
Main application entry point
"""

import os
import sys
from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'audio'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from app.main_screen import MainScreen
from app.training_screen import TrainingScreen
from app.model_viewer_screen import ModelViewerScreen
from app.settings_screen import SettingsScreen


class SonitApp(App):
    """Main Sonit application class"""
    
    title = "Sonit - Translating the Unspoken"
    icon = "logo.png"
    
    def build(self):
        """Build the application UI"""
        # Set window size for desktop
        if sys.platform != 'darwin':  # Not macOS
            Window.size = (800, 600)
        
        # Create screen manager
        sm = ScreenManager()
        
        # Add screens
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(TrainingScreen(name='training'))
        sm.add_widget(ModelViewerScreen(name='model_viewer'))
        sm.add_widget(SettingsScreen(name='settings'))
        
        return sm
    
    def on_start(self):
        """Called when the application starts"""
        print("Sonit started successfully!")
        print("Ready to translate vocal gestures...")
    
    def on_stop(self):
        """Called when the application stops"""
        print("Sonit shutting down...")


if __name__ == '__main__':
    try:
        SonitApp().run()
    except KeyboardInterrupt:
        print("\nSonit stopped by user")
    except Exception as e:
        print(f"Error starting Sonit: {e}")
        sys.exit(1) 