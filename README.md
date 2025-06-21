# Sonit

**Translating the Unspoken.**

Sonit is an open-source translator for non-verbal vocal gestures such as murmurs, hums, and culturally meaningful sounds (e.g., “hıh”, “ıhı”, “tsk”). It aims to give a voice to those who cannot speak, by interpreting subtle audio cues and translating them into words, intentions, or actions.

---

## 🧠 Overview

Sonit bridges the gap between vocal expression and spoken language. It’s designed for individuals with aphonia, neurological conditions, or temporary vocal loss, and for researchers working on human–machine interaction with minimal audio signals.

Sonit learns how a user expresses meaning through sound — then builds a personalized model to translate those expressions.

---

## 🔧 Tech Stack

- **Python** — Core logic, training pipeline
- **Kivy** — Lightweight cross-platform GUI (mobile + desktop)
- **PyTorch** — Deep learning model for sound classification
- **NumPy, Librosa** — Audio signal processing
- **SQLite** — Local user-specific training data

---

## 🔍 Features

- 🎙️ **Sound Input** — Captures vocal gestures like “mur”, “tsk”, “uhh”
- 🧬 **Sound-to-Intent Model** — Learns how each user expresses approval, refusal, interest, etc.
- 🧠 **Training Mode** — User/caregiver can label sounds and build a unique translation set
- 🧾 **Live Translation** — Real-time feedback showing interpreted meaning
- 📊 **Model Viewer** — See sound embeddings, confidence levels, and live output

---

## 💡 Use Cases

- Patients with vocal impairments (ALS, trauma, surgery recovery)
- Non-verbal communication training
- Assistive technology prototypes
- Research in minimal-signal communication

---

## 📁 Repository Structure

```

Sonit/
├── app/                # Kivy app UI
├── model/              # AI models & training
├── audio/              # Input, recording, analysis
├── data/               # User datasets
├── utils/              # Helper functions
├── main.py             # App entry point
└── README.md

````

---

## 🚀 Getting Started

```bash
git clone https://github.com/makalin/Sonit.git
cd Sonit
pip install -r requirements.txt
python main.py
````

**Dependencies**: Python 3.9+, PyTorch, Kivy, Librosa, NumPy, SoundFile

---

## 📈 Roadmap

* [x] Real-time sound capture
* [x] Labeling & training with minimal sounds
* [ ] Context-based intent prediction
* [ ] Model export/import across devices
* [ ] Caregiver override and suggestions
* [ ] Turkish sound gesture model (e.g., "ıhı", "hıh", "hımm")

---

## 🧪 Research

We're developing a small open dataset of Turkish vocal gestures. If you're a linguist, clinical expert, or accessibility researcher, get in touch!

---

## 📄 License

MIT License — free for personal, academic, and commercial use.

---

## 🙋‍♀️ Contributing

1. Fork the repo
2. Run the dev version and test with your own sounds
3. Submit improvements, bugfixes, or models via PR

We welcome help from:

* Accessibility advocates
* Audio researchers
* Speech therapists
* UI/UX designers for assistive apps

---

## 📬 Contact

> GitHub: [github.com/makalin/sonit](https://github.com/makalin/sonit)
> Email: [makalin@gmail.com](mailto:makalin@gmail.com)

---

> *“Not every voice speaks in words. Sonit listens anyway.”*
