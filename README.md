# IELTS Essay Generation and Evaluation Framework

This project provides an end-to-end pipeline for generating and evaluating IELTS Writing Task 2 essays using fine-tuned language models. It consists of three core components:
- A **GPT-Neo 125M-based generator** an decoder-based model for essay creation, because the small size and easier to train on limited hardware resource but provide a well performance
- A **BERT-based evaluator** a encoder model for scoring essays,
- A **T5-small-based model** using an encoder-decoder model to make this task as Seq2Seq task, with encoder will analyse the essay and decoder to generate an evaluation with detailed evaluation of IELTS writing components (Task Response, Coherence & Cohesion, Lexical Resource, Grammatical Range & Accuracy).

---

## Folder Structure
TEST_FOR_RUNNING_PROJECT/
├── notebook_finetune/ # All Jupyter notebooks for training & testing
│ ├── bert-finetune-for-evaluate-ielts-score.ipynb
│ ├── finetune_gpt_neo_for_generating_essay.ipynb
│ ├── generating-an-ielts-writing-task-2-evaluation.ipynb
├── test_for_evaluate_essay.ipynb # Evaluate real essays using the BERT models
├── test_for_evaluation_text.ipynb # Test for generating detailed evaluation for an essay based on band-score and question
├── test_for_model.ipynb # Test for generating essays from GPT-NEO fine-tuned model

---

##  Model Descriptions

###  GPT-Neo for Essay Generation
- **Checkpoint:** `gpt_ielts/checkpoint-8170`
- **Notebook:** `finetune_gpt_neo_for_generating_essay.ipynb`
- **New notebook:** `generating-ielts-writing-task-2-sample.ipynb`
- Fine-tuned on high-quality IELTS samples to generate Band 6–9 level essays based on input prompts/questions.
- New notebook that fine tune in processed dataset about consistency and more diverse field in IELTS Writing Task 2

###  BERT for Band Score Prediction
- **Checkpoint Directory:** `eval_score/`
- **Notebook:** `bert-finetune-for-evaluate-ielts-score.ipynb`
- Predicts overall band score (classification task) for a given essay. Trained using cross-entropy loss on labeled band score datasets.

### T5-small for Detailed IELTS Scoring (TR, CC, LR, GR)
- **Checkpoint:** `checkpoint-12300/`
- **Notebook:** `generating-an-ielts-writing-task-2-evaluation.ipynb`
- Generates scores for each individual IELTS writing component by leveraging sequence-to-sequence evaluation.

---

## Notebook Guide

| Notebook Name                                               | Purpose                                                                 |
|-------------------------------------------------------------|-------------------------------------------------------------------------|
| `finetune_gpt_neo_for_generating_essay.ipynb`               | Fine-tune GPT-Neo on IELTS Task 2 prompts and generate sample essays.  |
| `bert-finetune-for-evaluate-ielts-score.ipynb`              | Fine-tune BERT to classify essays into IELTS band scores.              |
| `generating-an-ielts-writing-task-2-evaluation.ipynb`       | Use T5-small to assess essays on four IELTS criteria.                  |
| `test_for_evaluate_essay.ipynb`                             | Evaluate a set of essays using both BERT and T5 models.                |
| `test_for_evaluation_text.ipynb`                            | Evaluate a single essay or free-text input for scoring.                |
| `test_for_model.ipynb`                                      | Utility notebook for loading models and debugging integration.         |

---

## Evaluation Strategy

- **BERT**: Classification accuracy and F1-score on band score predictions.
- **T5-small**: CrossEntropyLoss based on sample evaluation essay from dataset.
- **GPT-NEO finetune model**: Using some websibe that automated evaluate IELTS Writing score for testing performance of model.

---
## Requirement
- **Pytorch**: 
- **Transformers**: For loading pretrained model and tokenizer
- **Uvicorn**: For running API server
- **FastAPI**: for initialize application and setting body request.
- Command to install: pip install fastapi uvicorn transformers torch

---
## How to run
You need to run this command in terminal to loading app and activate the server
uvicorn app:app --host 0.0.0.0 --port 8000


