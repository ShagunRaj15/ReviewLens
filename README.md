# ReviewLens — Sentiment Intelligence Dashboard

ReviewLens is a Streamlit application for exploring sentiment in individual text and batches of customer reviews.

## What is included

- A redesigned dark analytics dashboard.
- Single text analysis with TXT/PDF upload.
- CSV batch analysis with a selectable review-text column.
- Sentiment distribution visualization.
- Downloadable enriched CSV and summary JSON.
- Session analysis history.
- Model evaluation page for saved training metrics.
- BERT inference with overlapping windows for longer documents.

## Important project and attribution note

This package was created as a substantial adaptation of the supplied `SentiBERT` starter project. The BERT/TF-IDF training pipeline and shared preprocessing helpers are retained from that starter. The ReviewLens interface and added workflows are a new implementation. Before submitting, make sure your course rules permit adapting this starter, document the work you personally completed, and acknowledge any source code or assistance your instructor requires you to cite. Do not claim the retained training code as independently written.

## Requirements

- Python 3.10 or newer recommended
- Internet connection for the first download of the pretrained model
- Optional GPU for faster training; CPU inference is possible but slower

## Run locally (Windows)

Open Command Prompt or the VS Code terminal in this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The browser should open the ReviewLens app. If it does not, open the local URL printed in the terminal.

## Run on Google Colab

This app is designed to run as a local Streamlit app. For Colab, install the requirements and use a Streamlit tunnelling method approved by your environment, or run it locally in VS Code.

## Train your own models (optional)

The app works with a public pretrained sentiment model if `models/bert` is not present. To train a local model:

```bash
python train_baseline.py
python train_bert.py --train_samples 5000 --epochs 2
```

Training writes model files and evaluation metrics into `models/`. Training BERT is resource-intensive; use a GPU runtime if available.

## CSV format

A CSV can contain any column name for the review text. Upload it in **Batch review lab**, then choose the appropriate column. The app analyzes up to 500 rows per run; begin with a small number to test performance.

Example:

| review |
|---|
| The delivery was quick and the quality was excellent. |
| The product stopped working after one day. |

## Interpretation and limitations

- The default pretrained model is trained for binary positive/negative sentiment on movie-review-style text.
- “Neutral / Mixed” is a heuristic label for scores near 0.5; it is not a separately trained third class.
- Domain shifts, sarcasm, irony, mixed opinions, short texts and non-English text can reduce reliability.
- Confidence is a model score, not a guarantee that a prediction is correct.
- Session history is temporary and is not stored in a database.
- Batch analysis runs each review individually and may take time for larger files.

## Suggested next development steps

1. Add unit tests for text cleaning and sentiment label mapping.
2. Add a persistent database for analysis history.
3. Evaluate on a labelled customer-review dataset and report measured metrics.
4. Add explainability and multilingual support only after testing them.
