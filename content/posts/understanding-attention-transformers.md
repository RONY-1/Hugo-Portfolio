---
title: "Understanding Attention Mechanisms in Vision Transformers"
date: 2025-08-05
draft: false
tags: ["Tutorial", "Deep Learning", "Vision Transformers"]
categories: ["Posts"]
summary: "A comprehensive deep dive into self-attention, patch embeddings, and multi-head attention visual explanations for computer vision."
showToc: true
---

### Introduction
Vision Transformers (ViT) have revolutionized computer vision by replacing traditional convolutional layers with multi-head self-attention mechanisms. In this post, we unpack how patches are extracted and processed.

### Patch Tokenization
Given an image $X \in \mathbb{R}^{H \times W \times C}$, we flatten it into 2D patches $X_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $(P, P)$ is the resolution of each image patch.

$$N = \frac{HW}{P^2}$$

### Self-Attention Calculation
The Query ($Q$), Key ($K$), and Value ($V$) projections are computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

This allows the model to compute global pairwise patch dependencies across the entire image.
