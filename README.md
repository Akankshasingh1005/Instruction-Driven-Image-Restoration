# Instruction-Driven Image Restoration (InstructIR Extensions)

A modular pipeline for **text-guided image restoration**, built on top of the InstructIR framework.
This project enables restoring images using simple natural language instructions instead of manually selecting task-specific models.

---

## What This Project Does

Traditional pipelines require you to choose a model for each task (denoising, deblurring, etc.).
Here, you can simply write:

> *"Remove noise and then enhance the colors"*

and the system will handle everything.

This repository focuses on extending InstructIR with **better control, interpretability, and usability**.

---

## Key Contributions

* **Multi-Step Instruction Handling**
  Breaks complex instructions into sequential steps for improved results.

* **Region-Aware Restoration**
  Applies restoration only to specific regions (e.g., *top*, *sky*, *right side*).

* **Confidence-Aware Outputs**
  Provides a confidence score and pixel-level confidence map.

* **Instruction Faithfulness Score (IFS)**
  Evaluates how well the output matches the given instruction.

---

## Setup 

### 1. Clone the original InstructIR repository

```bash id="xkq91m"
git clone https://github.com/mv-lab/InstructIR.git
cd InstructIR
```

---

### 2. Add this project’s code

Copy the `src/` folder from this repository after the cloned directory:

```id="zq2c1a"
InstructIR/
src/   ← paste here
```

---

### 3. Install dependencies

```bash id="c81l2p"
pip install scipy scikit-image ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

---

### 4. Run the notebook

Open and execute:

```id="m7xp9e"
notebook.ipynb
```

Make sure:

* Model weights are present in `models/`
* You are using GPU (Colab recommended)

---

## How It Works

1. Input image + natural language instruction
2. Instruction → text embedding
3. Embedding conditions the InstructIR model
4. Restoration is applied
5. Optional modules refine the output:

   * Multi-step processing
   * Region masking
   * Confidence estimation
   * Faithfulness scoring

---

## Project Structure

```id="h2q9aa"
src/
├── pipeline.py              # Main pipeline
├── instruction_parser.py   # Multi-step handling
├── region_selector.py      # Region masking
├── confidence_estimator.py # Confidence scoring
├── faithfulness_metric.py  # IFS computation
├── metrics.py              # PSNR, SSIM

notebook.ipynb
report.pdf
```

---

## Key Observations

* Sequential instructions outperform single combined prompts
* Region-aware restoration avoids unnecessary global changes
* Clear instructions produce more meaningful results
* PSNR/SSIM alone are not sufficient → IFS is needed

---

## Limitations

* Depends on pretrained InstructIR model
* Region selection is heuristic-based
* Limited evaluation dataset
* IFS is not fully aligned with human judgment

---

## Future Work

* Improve region detection using segmentation models
* Better instruction understanding using LLMs
* Larger real-world evaluation
* Improved faithfulness metrics

---

## Important Note

This repository contains only:

* `src/` (our extensions)
* `notebook.ipynb`

You must clone the original InstructIR repository before running.

---

## Project Info

Developed as part of a **Computer Vision project**
Focused on **instruction-driven and controllable image restoration**

---
## Report

For a detailed explanation of the methodology, experiments, and results,  
please refer to the full project report:

`report.pdf`
