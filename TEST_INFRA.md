# Test Infrastructure: SBC Scientific Paper E2E Suite

## Overview
This document specifies the architecture, operational execution, and verification criteria of the End-to-End (E2E) Test Suite for the SBC Scientific Paper on PNI vaccination microdata and AI pipeline (`tests/test_e2e_article.py` and `tests/run_e2e_tests.sh`).

The test suite provides an opaque-box, deterministic, automated verification harness covering all 14 features (F1 to F14), boundary constraints (strict 6 to 8 page budget), cross-module data consistency, and real-world compilation via Tectonic.

---

## 1. Test Architecture & Tier Mapping

The suite is structured across 4 distinct verification tiers in `tests/test_e2e_article.py`:

```
tests/test_e2e_article.py
├── TestTier1FeatureCoverage (14 tests: F1 to F14)
├── TestTier2BoundaryAndCornerCases (4 tests)
├── TestTier3CrossFeatureIntegration (2 tests)
└── TestTier4RealWorldWorkloads (3 tests)
```

### Tier 1 — Feature Coverage (F1 to F14)
* **F1 (`test_f1_sbc_template_and_metadata`):** Verifies existence of official SBC style files (`sbc-template.sty`, `sbc.bst`), `artigo.tex`, author/affiliation placeholders, and presence of both `resumo` and `abstract` environments conforming to <= 10 lines.
* **F2 (`test_f2_pni_public_health_context`):** Validates domain contextualization of public health (PNI, RNDS, DATASUS), citation of the 15,403 sanitized records, and definition of the `atraso_critico` target variable (> 7 days).
* **F3 (`test_f3_pipeline_integration_diagram`):** Verifies presence of the schematic integration diagram (`figures/pipeline_integrado.pdf` or `.png`) with size > 1KB and its inclusion in `artigo.tex` via `\includegraphics`.
* **F4 (`test_f4_decision_tree_id3_manual_entropy_info_gain`):** Asserts ID3 decision tree induction coverage, explicit mathematical formulations for Shannon Entropy $H(S)$ and Information Gain $IG(S, A)$, manual calculations for the first 3 levels (root entropy 0.9993, root winner `sistema_offline` IG 0.2133, Level 2 online/offline partitions, Level 3 branches), and 5-fold CV accuracy (~76.53%).
* **F5 (`test_f5_non_monotonicity_theoretical_discussion`):** Validates theoretical discussion (~0.5 page, >= 150 words) contrasting First-Order Logic (monotonicity) with Inductive Learning (non-monotonicity and refutability).
* **F6 (`test_f6_linear_models_ols_gd_ridge_logistic`):** Asserts OLS closed-form formulas ($\beta_1, \beta_0$), multivariate Gradient Descent ($J(\mathbf{w})$ and update rule), Ridge regression regularized objective ($J_{\text{Ridge}}$ with L2 penalty $\lambda$) and coefficient trajectory figure (`figures/etapa2_ridge_trajetoria`), and Logistic Regression ($\sigma(z)$, cross-entropy, ~77.47% accuracy).
* **F7 (`test_f7_probability_theory`):** Verifies joint distribution table of 2 binary variables with 4 atomic events, Kolmogorov axiomatic sum = 1.000000, marginals, conditional probabilities, Relative Risk ($3.17\times$), Chain Rule for 3 variables, Bayes' Theorem diagnostic inference ($76.91\%$), and conditional independence verification.
* **F8 (`test_f8_probabilistic_ml_and_em`):** Asserts MLE ($\hat{\theta}_{\text{MLE}} = 0.51591$) and MAP with conjugate Beta(2,2) prior ($\hat{\theta}_{\text{MAP}} = 0.51590$), Categorical Naive Bayes with Laplace smoothing (~75.79%), Expectation-Maximization (EM) algorithm for latent variable $Z$ with Gaussian Mixture, convergence figure (`figures/etapa4_em_convergencia`), and performance jump to $93.77\%$ accuracy.
* **F9 (`test_f9_bayesian_network`):** Verifies 5-node causal DAG ($R \to S \to A \leftarrow N \leftarrow V$), CPTs, exact inference by enumeration formula and result ($P(A=1|S=1, N=1) = 0.7458$), formal d-separation analysis (chain blocking and collider activation / explaining away), parameter economy (31 vs 10 parameters, $67.74\%$ reduction), and DAG figure (`figures/etapa5_rede_bayesiana_dag`).
* **F10 (`test_f10_fuzzy_logic`):** Asserts Mamdani fuzzy system with 2 antecedents (Bayesian Risk, Population Vulnerability) and 1 consequent (Intervention Urgency), triangular membership functions, 5 explicit IF-THEN rules, defuzzification via Centroid (CoG), Stage 5 posterior feeding Stage 6, comparison with hard threshold (0.5), and fuzzy surface figure (`figures/etapa6_fuzzy_superficie`).
* **F11 (`test_f11_bibtex_citations_and_appendix`):** Asserts `referencias.bib` existence, SBC-compliant citations (Russell & Norvig, Mitchell/Bishop, Zadeh, Ministério da Saúde/PNI), zero missing citations, and Methodology/Prompts Appendix in `artigo.tex`.
* **F12 (`test_f12_high_resolution_figures_present`):** Asserts all 5 required high-resolution figures exist in `figures/` with valid size (> 1000 bytes).
* **F13 (`test_f13_compilation_and_page_budget`):** Asserts `artigo.pdf` exists and page count is strictly between 6 and 8 pages inclusive.
* **F14 (`test_f14_git_commit_and_push_readiness`):** Asserts Git repository presence, active branch `main`, and configured remote `origin`.

---

### Tier 2 — Boundary & Corner Cases
* **Strict Page Budget Boundary (`test_strict_page_budget_boundary_enforcement`):** Verifies that the page evaluation logic strictly accepts 6, 7, and 8 pages, while unconditionally rejecting 5 pages (underflow) and 9 pages (overflow). If `artigo.pdf` exists, verifies its real page count conforms to this boundary.
* **BibTeX Validity & Zero Undefined References (`test_bibtex_validity_and_zero_undefined_references`):** Extracts all `\cite{...}` keys from `artigo.tex` and ensures 100% exist in `referencias.bib`. Checks `artigo.log` for any undefined citation warnings.
* **Image Format & Integrity (`test_image_format_and_integrity`):** Validates magic bytes for PDF vector figures (`%PDF-`) and tests PNG figures with Pillow (`Image.verify()`, minimum width >= 800px).
* **Resumo & Abstract Line Budget (`test_resumo_and_abstract_line_budget_boundary`):** Ensures neither Portuguese Resumo nor English Abstract exceeds 10 lines.

---

### Tier 3 — Cross-Feature Integration
* **Bayesian-to-Fuzzy Quantitative Coupling (`test_bayes_to_fuzzy_quantitative_coupling`):** Asserts that the exact posterior probabilities derived from Stage 5 enumeration feed into the Stage 6 Fuzzy system across the 3 SUS scenarios:
  - Rural Post: $P(A=1 \mid R=1, S=1, N=1) = 0.7458 \to \text{Urgency } 64.61$
  - Urban Unit: $P(A=1 \mid R=0, S=0, N=0) = 0.3059 \to \text{Urgency } 38.79$
  - Frontier Unit: $P(A=1 \mid R=1, S=0, N=1) = 0.1758 \to \text{Urgency } 23.90$
* **Numerical Consistency with Ground Truth (`test_numerical_claims_consistency_with_ground_truth`):** Verifies quantitative consistency across text and tables:
  - 15,403 samples
  - Root entropy: 0.9993 bits
  - Decision tree 5-fold CV accuracy: 76.53%
  - Logistic regression accuracy: 77.47%
  - Kolmogorov sum: 1.000000
  - Relative risk: 3.17x
  - Naive Bayes jump: 75.79% -> 93.77%
  - Bayesian Network parameter savings: 67.74% (31 vs 10)

---

### Tier 4 — Real-World Workload Scenarios
* **Tectonic Binary Check (`test_tectonic_binary_availability`):** Verifies `/home/luizg/.local/bin/tectonic` executable presence and version response.
* **Compilation Execution (`test_tectonic_compilation_execution`):** Executes full headless LaTeX compilation (`tectonic artigo.tex`), verifying zero fatal errors, package downloads, and clean output.
* **Clean Build Verification (`test_clean_build_verification`):** Validates generated `artigo.pdf` size and page count compliance.

---

## 2. Test Execution Instructions

### Complete Suite Execution (Default)
```bash
# Via shell runner
./tests/run_e2e_tests.sh

# Or directly via Python unittest
.venv/bin/python -m unittest -v tests/test_e2e_article.py
```

### Running Specific Tiers
```bash
# Tier 1 only (Feature Coverage)
./tests/run_e2e_tests.sh -t 1

# Tier 2 only (Boundary & Corner Cases)
./tests/run_e2e_tests.sh -t 2

# Tier 3 only (Cross-Feature Integration)
./tests/run_e2e_tests.sh -t 3

# Tier 4 only (Real-World Workloads)
./tests/run_e2e_tests.sh -t 4
```

### Exit Codes
* `0`: All executed test cases passed successfully.
* `1`: One or more test cases failed (expected prior to M1, M2, and M3 completion).

---

## 3. Environment Dependencies

| Tool / Library | Path / Requirement | Purpose |
|---|---|---|
| **Python** | `/home/luizg/trabalho_fia_final/.venv/bin/python` (Python 3.13.5) | Test runtime engine |
| **unittest** | Standard library | Test runner and assertions |
| **Pillow** | Installed in `.venv` (v13.0.0) | Image format and resolution verification |
| **Tectonic** | `/home/luizg/.local/bin/tectonic` (v0.17.0) | Complete standalone LaTeX/BibTeX compiler |
| **pdfinfo** | `/usr/bin/pdfinfo` | Authoritative PDF metadata and page count extractor |
| **Git** | `/usr/bin/git` | Repository status and branch validation |
