# DriftWatchdog

Watches a real, already-deployed classifier for data drift - because most
ML projects stop at training a model and never ask how you'd notice it
silently decaying in production.

## Screenshots

![Home](screenshots/home.png)
![Watchdog](screenshots/watchdog.png)
![Timeline](screenshots/timeline.png)

## How it works

- Wraps the real MLP emotion classifier and its 180-feature training
  distribution (reused from EmotionDetect, a sibling project).
- Injects synthetic production drift - a sensor change, added noise, a
  shifting population, a broken feature - at an adjustable intensity.
- Detects it with a per-feature Kolmogorov-Smirnov test, Bonferroni-corrected
  across all 180 features so it isn't flagging drift by chance.
- Tracks the model's real accuracy on the same drifted batch, so the
  "Timeline" view shows the drift signal and actual accuracy decay moving
  together, not two independently faked numbers.

## Running it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT, see `LICENSE`. Statistical drift detection via ml3-drift (Apache-2.0).
