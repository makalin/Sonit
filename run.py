#!/usr/bin/env python3
"""
Run script for Sonit application
Simple entry point for running the application
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point"""
    try:
        # Import and run the main application
        from main import SonitApp
        app = SonitApp()
        app.run()
    except KeyboardInterrupt:
        print("\nSonit stopped by user")
    except Exception as e:
        print(f"Error running Sonit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 