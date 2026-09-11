# Transformer from Scratch — Sentiment Analysis

A simple implementation of a Transformer-based sentiment analysis model using TensorFlow/Keras.

This project was built to understand the internal working of the Transformer architecture by implementing its major components step-by-step instead of directly using a pre-built Transformer model.

---

## 📌 Project Overview

The goal of this project is to understand how a Transformer Encoder processes text and uses self-attention to capture relationships between words.

The model performs binary sentiment classification:

- `1` → Positive
- `0` → Negative

The project uses a small custom dataset of 30 movie-review-style sentences.

---

## 🧠 Transformer Architecture

The complete pipeline is:

```text
Raw Text
   ↓
Tokenization
   ↓
Padding
   ↓
Word Embedding
   ↓
Positional Encoding
   ↓
Multi-Head Self-Attention
   ↓
Add & Norm
   ↓
Feed Forward Network
   ↓
Add & Norm
   ↓
Global Average Pooling
   ↓
Dense + Sigmoid
   ↓
Positive / Negative
