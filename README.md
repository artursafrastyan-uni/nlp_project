# Mechanistic Interpretability of GPT-2 Small: Circuits, Induction, and Redundancy

This repository contains the codebase for my final NLP project, focused on reverse-engineering the internal circuits of **GPT-2 Small (124M parameters)** using mechanistic interpretability techniques. 

By using the `TransformerLens` library to inspect, hook, and manipulate intermediate activations, this project replicates key findings from two foundational papers in the field:
1. **Olsson et al. (Anthropic)**: *"Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning"*
2. **Wang et al. (Redwood Research)**: *"Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 small"*

The complete project code, datasets, and LaTeX-formatted final paper are available in the GitHub repository: [artursafrastyan-uni/nlp_project](https://github.com/artursafrastyan-uni/nlp_project).

---

## Project Structure

* **`nlp_project_main.py`**: The primary execution script containing the implementation of all datasets, attention pattern analysis, direct logit attribution, activation patching, and zero-ablation experiments.
* **`nlp_project.ipynb` / `fixed_notebook.ipynb`**: Interactive Jupyter notebooks walking through the same experimental phases.
* **`synthetic_dataset.pt`**: Pre-generated synthetic repeated sequences ($A \rightarrow A$) used for induction head scoring.
* **`templated_dataset.pt`**: Pre-generated names and places templates used for the Indirect Object Identification (IOI) circuit evaluation.
* **`induction_heads.tex`**: The complete LaTeX paper summarizing the findings, LaTeX setups, and analysis.

---


### 1. Installation & Environment Setup
To run the code, make sure you have Python 3.8+ installed. You can set up a virtual environment and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/artursafrastyan-uni/nlp_project.git
cd nlp_project

# Set up and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install torch transformers transformer_lens nbformat python-docx
```

### 2. Running the Analysis
You can execute the main script to run the full pipeline, which generates the datasets (if not already cached), identifies induction/name-moving heads, runs activation patching, and performs zero-ablations:

```bash
python nlp_project_main.py
```

---

## Core Experiments & Discoveries

### Part 1: Induction Heads & In-Context Learning (ICL)
* **What are they?** Induction heads are attention heads that implement the pattern-matching rule: $[A][B] \dots [A] \rightarrow [B]$.
* **Attention Pattern Analysis**: By calculating prefix-matching scores on repeated synthetic sequences, we isolated middle-layer attention heads responsible for induction. The top induction head is **L5H5** (score: 0.9150), followed by **L5H1** and **L7H10**.
* **Direct Logit Attribution**: Bypassing the rest of the network via a "logit lens" projection revealed that L5H5 writes directly to the vocabulary, boosting the target token's logit by an average of **+4.7380**.
* **Causal Patching (Activation Patching)**: Patching L5H5 alone onto a corrupted sequence only recovered **+0.3942** logits. However, patching the full induction ensemble (L5H5, L5H1, L6H9, L7H10) recovered **+5.6712** logits. This proves that induction is a distributed, multi-head circuit operating as an ensemble.
* **Zero-Ablation**: Silencing the top 10 induction heads crashed the model's confidence on the sequence continuation task from **75.8% to 11.8%** (a **-64.0%** drop), compared to a negligible **-7.7%** drop when deactivating random control heads.

### Part 2: Indirect Object Identification (IOI) & Backup Heads
* **What is it?** The semantic task of predicting the indirect object in templated clauses (e.g. *"John and Mary went to the store. John gave a book to [Mary]"*).
* **Name Mover Identification**: We identified **L9H9** as a primary Name Mover Head, writing an average logit of **+0.6687** for the target name directly into the final token's residual stream.
* **Backup Head Discovery (Circuit Redundancy)**: To test the resilience of the network, we zero-ablated the three primary Name Movers (**L9H9, L9H6, L10H0**). The target logit only decreased from **4.7316 to 4.4870** (a minor **-0.2446** drop). This confirms the existence of *Backup Name Mover Heads* (such as L10H7) that dynamically increase their activation strength via LayerNorm calibration and active compensation to safeguard the model's performance.

---

## Citation & Reference
If you want to read the full details of this research, please check the [LaTeX paper](induction_heads.tex) included in the repo.

* **Author**: Artur Safrastyan ([artur.safrastyan@email.com](mailto:artur.safrastyan@email.com))
* **Git Repository**: [https://github.com/artursafrastyan-uni/nlp_project](https://github.com/artursafrastyan-uni/nlp_project)
