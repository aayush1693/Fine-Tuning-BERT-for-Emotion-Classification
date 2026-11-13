# BERT Fine-Tuning for Emotion Classification

Fine-tunes `bert-base-uncased` on the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset to classify English text into six emotions: sadness, joy, love, anger, fear and surprise. The notebook compares the fine-tuned model against four baselines on the same test set.

## Results

Test set: 2,000 examples. Fine-tuning results are mean ± std over three seeds (42, 43, 44).

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | 0.347 | 0.086 |
| Zero-shot NLI (`bart-large-mnli`) | 0.466 | 0.433 |
| Frozen BERT + linear probe | 0.590 | 0.530 |
| TF-IDF + logistic regression | 0.882 | 0.837 |
| **BERT fine-tuned (3 seeds)** | **0.927 ± 0.003** | **0.884 ± 0.006** |

- Fine-tuned BERT beats the strongest baseline (TF-IDF + logistic regression) by about 5 points of macro-F1. A paired bootstrap gives a 95% CI of +0.027 to +0.077 for the gain, and an exact McNemar test gives p ≈ 4e-11 (155 examples only BERT got right vs. 59 only the baseline got right). Both results are for seed 42.
- Frozen BERT features with a linear probe perform poorly. Most of the benefit comes from fine-tuning, not from the pre-trained representation alone.
- Per-class F1 (seed 42): sadness 0.96, joy 0.95, anger 0.92, fear 0.90, love 0.84, surprise 0.75. Surprise is the weakest class, with only 66 test examples.

## Method

- **Data:** the standard train/validation/test split (16,000 / 2,000 / 2,000). The label distribution is skewed toward joy and sadness, so macro-F1 is reported alongside accuracy.
- **Model:** `bert-base-uncased` with a sequence-classification head. Max sequence length 128.
- **Training:** 3 epochs, learning rate 2e-5, batch size 32, weight decay 0.01, 6% warm-up, FP16 on GPU.
- **Model selection:** the checkpoint with the best validation macro-F1 is kept. The test set is used only once per seed, after training.
- **Efficiency:** 109M parameters (418 MB in FP32). About 13 ms per example at batch size 1 and about 565 examples/s at batch size 32 on a Tesla T4.

## Requirements

- A GPU runtime (developed on a Colab T4; the notebook falls back to CPU but training will be slow)
- Python packages: `transformers`, `datasets`, `accelerate`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `scipy`
- Optional: a Hugging Face token in the `HF_TOKEN` environment variable or Colab secrets. Needed for the zero-shot baseline download.

## Usage

Open `bert_emotion_classification.ipynb` in Google Colab with a GPU runtime and run all cells. The notebook downloads the dataset and models automatically and writes results to `/content/bert_out/`:

- `leaderboard.csv`
- `efficiency.json`
- `data_overview.png`
- `validation_curves.png`
- `comparison.png`
- `confusion_matrix.png`
- the best seed-42 model

The zero-shot baseline downloads `facebook/bart-large-mnli` (about 1.6 GB). Set `RUN_ZEROSHOT = False` in the configuration cell to skip it.

## Limitations

- Results come from one train/validation/test split and three seeds.
- No hyperparameter search was done. The configuration is a standard BERT fine-tuning recipe.
- The dataset labels were assigned using hashtags, so some errors reflect label noise. Many of the most confident mistakes involve mixed or ambiguous emotions.
- Results apply to this dataset and may not transfer to other domains such as social media or longer documents.
