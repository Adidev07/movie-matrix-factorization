# Movie Recommendation with Matrix Factorization

Learn latent user and item factors from sparse ratings and evaluate both rating error and held-out ranking.

Actual model fitting with NumPy. Synthetic data; educational benchmark, not production performance.

## Quick start

Python 3.10+ required.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python project.py
python -m unittest discover -v
```

Open `results/report.html` after running. View [saved metrics](results/metrics.json), [PDF guide](PROJECT_GUIDE.pdf), or [editable guide](GUIDE.md).

## Experiment

Use 1000 training pairs, 250 validation pairs and 250 test pairs. Compare models trained for 40 versus 100 epochs using validation RMSE. Evaluate the selected model on test without retraining. Rating baseline is the global training mean. For ranking, exclude training and validation items for each user, then compare top five recommendations with test ratings at least 3.5. Macro recall averages only users with at least one relevant test item. Compare with a training-item-mean ranking baseline.

## What to explain in an interview

Why factorization? It shares information across sparse user-item interactions using a small latent space. Why RMSE and recall? Rating accuracy and ranking usefulness measure different objectives. What is cold start? A new user or item has no learned factor. This implementation supports only known IDs and does not implement cold-start fallback. Random splitting is optimistic for future recommendations. Extend with a licensed public ratings dataset, chronological splits, popularity and content fallbacks, diversity checks, group-aware coverage tests and implicit-feedback ranking objectives. Treat unobserved items as unknown rather than automatically disliked.

## Files

- `project.py`: training, inference and experiment.
- `common.py`: metrics and HTML/JSON report output.
- `test_project.py`: four behavioral tests.
- `results/`: model, data and measured experiment outputs.
- `BUILD_INFO.txt`: verification environment.

## Attribution and limitations

AI-assisted learning project. Review and customize before presenting your contributions. Uses synthetic data and deliberately limited models. No external model API or personal data. MIT license.
