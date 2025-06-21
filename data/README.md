# Data Directory

This directory contains all data files for the Sonit application.

## Structure

- `recordings/` - Audio recordings from training sessions
- `models/` - Trained model files
- `plots/` - Generated plots and visualizations
- `sonit.db` - SQLite database with samples and metadata

## Files

- Audio files (`.wav`) are automatically saved here during training
- Model files (`.pth`, `.pkl`) are saved here after training
- Database file contains all samples, models, and training history

## Backup

The database and important files should be backed up regularly, especially before major updates. 