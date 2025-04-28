# AI-Powered CV–Job Matching System (LLM Enhanced)

> 🧠🔍 **DevMatch**: Reimagining how you connect brilliant developers to the right roles – lightning-fast, hyper-accurate, and powered by cutting-edge AI.

---

## 🚀 Table of Contents
1. [What Is DevMatch?](#what-is-devmatch)
2. [Key Features](#key-features)
3. [Architecture Overview](#architecture-overview)
4. [Installation & Quickstart](#installation--quickstart)
5. [Usage Examples](#usage-examples)
6. [Scoring Methodology](#scoring-methodology)
7. [Configuration & Tuning](#configuration--tuning)
8. [Tech Stack](#tech-stack)
9. [Contributing](#contributing)
10. [License](#license)

---

## 🎯 What Is DevMatch?
DevMatch is an **AI-powered CV–Job matching engine** that bridges the gap between project requirements and developer profiles. Leveraging large language models and semantic embeddings, DevMatch delivers:

- 🔹 **Top-5 candidate recommendations** in under **3 seconds**  
- 🔹 **Transparent scoring** with detailed explanations  
- 🔹 **Industry-aware matches** (banking, healthcare, fintech…)  
- 🔹 **Customizable skill weighting** per job  

Whether you’re HR, a hiring manager, or a recruitment platform, DevMatch turns the manual matchmaking pain into an automated, scalable delight.

---

## ✨ Key Features

| Feature                                 | Description                                                  |
|-----------------------------------------|--------------------------------------------------------------|
| **Semantic Matching (60%)**             | Deep cosine similarity via SBERT/JobBERT embeddings          |
| **Skill-Weighted Matching (30%)**       | Predefined technical skills with user-assigned weights       |
| **Industry Knowledge Scoring (10%)**    | Regex & ML-driven detection of domain-specific experience    |
| **Lightning-Fast Indexing**             | FAISS index for sub-second nearest neighbor search           |
| **Async Explanation Generation**        | LLM-powered natural language rationale for each match        |
| **Dual Interfaces**                     | 1. Job → CVs & 2. CV → Jobs                                  |
| **Configurable & Extensible**           | Add new industries, tweak weights, swap embedder or LLM      |

---
