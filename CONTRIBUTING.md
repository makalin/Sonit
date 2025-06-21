# Contributing to Sonit

Thank you for your interest in contributing to Sonit! This document provides guidelines for contributing to the project.

## 🎯 Project Goals

Sonit aims to:
- Provide accessible communication tools for people with vocal impairments
- Advance research in minimal-signal communication
- Create an open-source platform for vocal gesture translation
- Support multiple languages and cultural sound gestures

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Git
- Basic knowledge of audio processing and machine learning

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/Sonit.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Install dev dependencies: `pip install -r requirements.txt[dev]`

## 📁 Project Structure

```
Sonit/
├── app/                    # Kivy GUI application
│   ├── main_screen.py     # Main translation interface
│   ├── training_screen.py # Training mode interface
│   ├── model_viewer_screen.py # Model visualization
│   └── settings_screen.py # Settings interface
├── audio/                 # Audio processing
│   ├── recorder.py        # Audio capture
│   └── processor.py       # Feature extraction
├── model/                 # Machine learning models
│   ├── predictor.py       # Sound prediction
│   └── trainer.py         # Model training
├── utils/                 # Utilities
│   ├── database.py        # Data storage
│   └── config.py          # Configuration
├── data/                  # Data storage
├── tests/                 # Test suite
└── main.py               # Application entry point
```

## 🧪 Testing

Run tests with pytest:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_audio.py
```

## 📝 Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **Type hints** for better code documentation

Format your code:
```bash
black .
```

Check for issues:
```bash
flake8 .
```

## 🔧 Development Workflow

1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes**: Follow the coding standards
3. **Add tests**: Include tests for new functionality
4. **Run tests**: Ensure all tests pass
5. **Commit your changes**: Use descriptive commit messages
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Create a pull request**: Provide a clear description of your changes

## 🎨 UI Development

When working on the GUI:
- Follow Kivy design patterns
- Test on different screen sizes
- Ensure accessibility features work
- Use the existing color scheme and styling

## 🔬 Research Contributions

We welcome contributions from:
- **Speech therapists**: Help improve training protocols
- **Linguists**: Contribute to sound gesture databases
- **Accessibility researchers**: Suggest improvements for users
- **Audio engineers**: Optimize audio processing

## 🐛 Bug Reports

When reporting bugs:
1. Use the GitHub issue template
2. Include steps to reproduce
3. Provide system information
4. Attach relevant logs or error messages

## 💡 Feature Requests

For feature requests:
1. Check existing issues first
2. Describe the use case clearly
3. Explain how it benefits users
4. Consider implementation complexity

## 📚 Documentation

Help improve documentation by:
- Adding docstrings to new functions
- Updating README.md with new features
- Creating tutorials for complex features
- Translating documentation to other languages

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help other contributors
- Share knowledge and resources
- Follow the project's code of conduct

## 🏷️ Release Process

1. **Version bump**: Update version in `setup.py` and `main.py`
2. **Changelog**: Update CHANGELOG.md with new features/fixes
3. **Tests**: Ensure all tests pass
4. **Documentation**: Update documentation if needed
5. **Tag release**: Create a git tag for the version
6. **Publish**: Release on GitHub

## 📞 Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact makalin@gmail.com for private matters

## 🙏 Acknowledgments

Thank you for contributing to making communication more accessible for everyone!

---

*"Not every voice speaks in words. Sonit listens anyway."* 