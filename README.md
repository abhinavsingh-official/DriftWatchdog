# DriftWatchdog

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

Watches a real, already-deployed classifier for data drift. It wraps a real
MLP classifier and its 180-feature training distribution, injects synthetic
production drift (a sensor change, added noise, a shifting population, a
broken feature) at an adjustable intensity, detects it with a per-feature
Kolmogorov-Smirnov test Bonferroni-corrected across all 180 features, and
tracks the model's real accuracy on the same drifted batch - so the
Timeline view shows the drift signal and the actual accuracy decay moving
together.

## Screenshots

![Home](screenshots/home.png)
![Watchdog](screenshots/watchdog.png)
![Timeline](screenshots/timeline.png)

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `python -m pytest tests/ -v`
