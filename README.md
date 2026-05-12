# LAH-TWGNN

This is the supplementary code repository for the paper

> **Logic-Aware Hierarchical Graph Neural Networks for First-Order Formula Relevance.**

Rather than treating premise selection as a flat pair-classification problem, LAH-TWGNN walks a formula pair through a small pipeline of *typed* graph operations and *logic-aware* corrections. The README below follows that pipeline: each section describes one stage, points to the file where it lives, and explains what to expect when you run it.

---

## What the model does, stage by stage

### Stage 1 — From a first-order formula to a typed term-walk graph

A formula is parsed by `formula_parser.py` (Lark grammar) into the `Graph` / `Node` structures defined in `graph.py`. Every node carries its connective or symbol type; every edge carries an argument-position label. The walks `T_u(v)`, `T_m(v)`, `T_l(v)` referenced in the paper are constructed in `model.py` inside `DAGEmbedding`, lines roughly 88–128.

### Stage 2 — Position-aware message passing

`DAGEmbedding` runs a small MLP per term-walk position (`F_T`, `F_M`, `F_B`) followed by a gated update — this is the part of the paper that justifies the qualifier *typed*: connective semantics are preserved instead of averaged out by an order-invariant aggregator.

### Stage 3 — Hierarchical attention pooling (HAP)

The `HierarchicalPooling` class in `model.py` realises Algorithm 1 of the paper. Subformulas are pooled in increasing order of depth using a node-level attention mechanism, then weighted by a salience gate `β_S`. The class is small enough (≈ 50 LOC) to read end-to-end if you want to map symbols back to the equations.

### Stage 4 — Symbolic priors injected into the logits

After pooling, the `Classifier` applies three soft rules — subformula similarity, semantic-symbol co-occurrence, quantifier-path overlap — together with two hard masks (a conservative tautology mask and a predicate-disjointness negative prior). These appear in lines 208–277 of `model.py`. The corresponding rule weights `α_r1, α_r2, α_r3` are exposed as command-line flags in `eval.py`.

### Stage 5 — Logic-aware training

The total loss is the binary cross-entropy on the relevance label combined with the rule-loss term defined in `LogicLoss`. Multi-task weighting is delegated to MGDA from `LibMTL.weighting`, matching the Pareto-consistent scheme analysed in the paper.

---

## Repository layout

```
supplementary material_LAH/
├── ATP_experiment/
│   ├── code/                 Core model + parser + dataset + classifier
│   ├── dataset/              Held-out problems for end-to-end proving
│   └── scripts/              feedback_loop.py, evaluate_*.py
└── premise_selection/
    ├── MPTP/                 Trains LAH-TWGNN on MPTP2078
    └── CNF/                  Same model on the clausified variant
```

The `code/` directory contains a number of artefacts that are easy to overlook on first glance: `select_premises_lah.py` is the inference-time premise ranker, `convert_data_for_lah.py` is the one-off data preparation script, and `node_dict.pkl` is the cached symbol vocabulary used by every script.

---

## Installation

```bash
pip install -r requirements.txt
```

A working PyTorch ≥ 1.10 (with CUDA recommended) is assumed. The only slightly unusual dependency is **LibMTL**, which provides the MGDA multi-task weighting referenced in stage 5; everything else is standard PyTorch / PyTorch Geometric tooling. The end-to-end stage 6 below additionally requires **E-prover** ≥ 2.6 on `$PATH`.

---

## Stage 6 — Running the model

### Premise-selection benchmarks

Both subdirectories follow the same pattern:

```bash
cd premise_selection/MPTP
python eval.py        # train + validate + test on MPTP2078

cd ../CNF
python eval.py        # same pipeline on the CNF counterpart
```

Default hyper-parameters reproduce the numbers in the paper: mean accuracy **88.83 %** on MPTP and **86.21 %** on CNF.

### End-to-end proving with E-prover

```bash
cd ATP_experiment/scripts
python feedback_loop.py        # iterative selection + proof attempts
python evaluate_simple.py      # single-pass ranking baseline
python evaluate_fixed_k.py     # fixed-k ablation
python evaluate_sine.py        # sInE comparison
python evaluate_cascade.py     # cascade retrieval
```

`feedback_loop.py` produces the **253 / 280** result on MPTP2078 reported in the abstract.

---

## A short note on what is *not* in the code

The paper proves a number of structural properties of the model — invariance, traceable salience, stable subformula matching, odds-ratio interpretability of the rule correction, Pareto-consistency of the loss weighting. These are theoretical results; the code does not contain experimental scripts for them. The relevant equations in `model.py` are however annotated with the property they realise, so a reader who wants to verify a claim numerically can do so by ablating the corresponding sub-module.

---

## Citation

If you build on this code please cite the accompanying paper. The BibTeX entry will be added once the work is officially published.
