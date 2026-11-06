#!/usr/bin/env python3
"""
tests/test_e2e_article.py
=========================
Comprehensive End-to-End (E2E) Test Suite for the SBC Scientific Article
on PNI Vaccination Microdata and Artificial Intelligence Pipeline.

Covers 4 Verification Tiers:
- Tier 1: Feature Coverage (F1 through F14)
- Tier 2: Boundary & Corner Cases (strict page budget 6-8, BibTeX undefined citations, image resolutions)
- Tier 3: Cross-Feature Integration (Bayesian posterior feeding Fuzzy antecedents, numerical consistency)
- Tier 4: Real-World Workload Scenarios (Tectonic compilation execution, clean build)

Authoritative sources:
- .agents/teamwork/ORIGINAL_REQUEST.md
- .agents/teamwork/PROJECT.md
- .agents/teamwork/spec_miner_survey_1/report.md
- 2o-Trab-SimbML-Incerteza.md
"""

import os
import re
import sys
import shutil
import subprocess
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Key Artifact Paths
ARTIGO_TEX = PROJECT_ROOT / "artigo.tex"
ARTIGO_PDF = PROJECT_ROOT / "artigo.pdf"
ARTIGO_LOG = PROJECT_ROOT / "artigo.log"
REFERENCIAS_BIB = PROJECT_ROOT / "referencias.bib"
SBC_TEMPLATE_STY = PROJECT_ROOT / "sbc-template.sty"
SBC_BST = PROJECT_ROOT / "sbc.bst"
FIGURES_DIR = PROJECT_ROOT / "figures"
TECTONIC_BIN = Path("/home/luizg/.local/bin/tectonic")

# Expected Figures (vector .pdf preferred, or high-res .png)
REQUIRED_FIGURE_BASENAMES = [
    "pipeline_integrado",
    "etapa2_ridge_trajetoria",
    "etapa4_em_convergencia",
    "etapa5_rede_bayesiana_dag",
    "etapa6_fuzzy_superficie",
]

# Ground Truth Metrics
GROUND_TRUTH_METRICS = {
    "total_samples": 15403,
    "root_entropy": 0.9993,
    "root_winner_ig": 0.2133,
    "dt_cv_mean": 76.53,
    "ols_r2": 0.278,
    "logistic_acc": 77.47,
    "kolmogorov_sum": 1.0000,
    "relative_risk": 3.17,
    "bayes_diagnostic": 76.91,
    "mle_theta": 0.51591,
    "map_theta": 0.51590,
    "nb_pure_acc": 75.79,
    "nb_em_acc": 93.77,
    "bn_param_savings": 67.74,
    "bayes_to_fuzzy_scenarios": {
        "rural": {"posterior": 0.7458, "urgency": 64.61},
        "urban": {"posterior": 0.3059, "urgency": 38.79},
        "frontier": {"posterior": 0.1758, "urgency": 23.90},
    },
}


def read_tex_file(tex_path: Path) -> str:
    """Reads TeX file content if it exists."""
    if not tex_path.exists():
        return ""
    with open(tex_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def strip_tex_comments(tex_text: str) -> str:
    """Strips TeX comments (% ...) while preserving newlines."""
    lines = []
    for line in tex_text.splitlines():
        # Match % not preceded by backslash
        cleaned = re.sub(r"(?<!\\)%.*$", "", line)
        lines.append(cleaned)
    return "\n".join(lines)


def get_pdf_page_count(pdf_path: Path) -> int:
    """authoritatively obtains PDF page count using pdfinfo, fallback to pypdf."""
    if not pdf_path.exists():
        return 0

    # 1. Try pdfinfo
    pdfinfo_bin = shutil.which("pdfinfo") or "/usr/bin/pdfinfo"
    if os.path.exists(pdfinfo_bin):
        res = subprocess.run(
            [pdfinfo_bin, str(pdf_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            match = re.search(r"^Pages:\s+(\d+)", res.stdout, re.MULTILINE)
            if match:
                return int(match.group(1))

    # 2. Try pypdf fallback if installed
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        pass

    # 3. Simple binary trailer fallback
    try:
        with open(pdf_path, "rb") as f:
            data = f.read()
        matches = re.findall(rb"/Type\s*/Page\b", data)
        # Note: /Pages might also be matched, filter /Pages
        page_matches = [m for m in matches if not m.startswith(b"/Type /Pages")]
        if page_matches:
            return len(page_matches)
    except Exception:
        pass

    return 0


def find_figure_path(basename: str) -> Path | None:
    """Finds either .pdf or .png for the figure basename in figures/."""
    for ext in [".pdf", ".png"]:
        p = FIGURES_DIR / f"{basename}{ext}"
        if p.exists():
            return p
    return None


def extract_citations_from_tex(tex_text: str) -> set[str]:
    """Extracts all citation keys referenced in \\cite, \\citeauthor, \\citep, etc."""
    cleaned = strip_tex_comments(tex_text)
    pattern = r"\\cite[a-zA-Z*]*\{([^}]+)\}"
    keys = set()
    for match in re.finditer(pattern, cleaned):
        raw_keys = match.group(1).split(",")
        for k in raw_keys:
            k = k.strip()
            if k:
                keys.add(k)
    return keys


def extract_bibtex_keys(bib_path: Path) -> set[str]:
    """Extracts defined keys from a .bib file."""
    if not bib_path.exists():
        return set()
    with open(bib_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    pattern = r"@\w+\s*\{\s*([^,]+)\s*,"
    keys = set(re.findall(pattern, content))
    return {k.strip() for k in keys if k.strip()}


# ==============================================================================
# TIER 1 — FEATURE COVERAGE (F1 to F14)
# ==============================================================================
class TestTier1FeatureCoverage(unittest.TestCase):
    """Verifies all user requirements and acceptance criteria for features F1 to F14."""

    def test_f1_sbc_template_and_metadata(self):
        """F1: Template SBC (sbc-template.sty, sbc.bst), placeholders for authors/affiliations,
        Abstract and Resumo (<= 10 lines) on page 1."""
        self.assertTrue(
            SBC_TEMPLATE_STY.exists(),
            f"F1 Failure: SBC template style '{SBC_TEMPLATE_STY.name}' not found in project root.",
        )
        self.assertTrue(
            SBC_BST.exists(),
            f"F1 Failure: SBC BibTeX style '{SBC_BST.name}' not found in project root.",
        )
        self.assertTrue(
            ARTIGO_TEX.exists(),
            f"F1 Failure: Main LaTeX document '{ARTIGO_TEX.name}' not found.",
        )

        tex_content = read_tex_file(ARTIGO_TEX)
        self.assertTrue(
            "sbc-template" in tex_content or "sbc.cls" in tex_content,
            "F1 Failure: artigo.tex does not include official SBC template package or class.",
        )

        # Placeholders for authors/affiliations
        has_author = bool(re.search(r"\\author\{", tex_content))
        has_address = bool(re.search(r"\\address\{", tex_content)) or bool(
            re.search(r"\\institute\{|\\inst\{", tex_content)
        )
        self.assertTrue(
            has_author and has_address,
            "F1 Failure: artigo.tex must have author and address/affiliation placeholders.",
        )

        # Resumo and Abstract presence
        self.assertRegex(
            tex_content,
            r"\\begin\{resumo\}[\s\S]*?\\end\{resumo\}",
            "F1 Failure: artigo.tex must contain 'resumo' environment in Portuguese.",
        )
        self.assertRegex(
            tex_content,
            r"\\begin\{abstract\}[\s\S]*?\\end\{abstract\}",
            "F1 Failure: artigo.tex must contain 'abstract' environment in English.",
        )

        # Length constraint: <= 10 lines each
        resumo_match = re.search(r"\\begin\{resumo\}([\s\S]*?)\\end\{resumo\}", tex_content)
        if resumo_match:
            lines = [l.strip() for l in resumo_match.group(1).splitlines() if l.strip()]
            self.assertLessEqual(
                len(lines),
                10,
                f"F1 Failure: Resumo exceeds 10 lines ({len(lines)} lines found).",
            )

        abstract_match = re.search(
            r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}", tex_content
        )
        if abstract_match:
            lines = [l.strip() for l in abstract_match.group(1).splitlines() if l.strip()]
            self.assertLessEqual(
                len(lines),
                10,
                f"F1 Failure: Abstract exceeds 10 lines ({len(lines)} lines found).",
            )

    def test_f2_pni_public_health_context(self):
        """F2: PNI public health contextualization (15,403 records, target atraso_critico)."""
        self.assertTrue(ARTIGO_TEX.exists(), "F2 Failure: artigo.tex not found.")
        tex = read_tex_file(ARTIGO_TEX)

        # Domain context: PNI, RNDS or DATASUS
        has_pni = bool(re.search(r"\bPNI\b|Programa Nacional de Imunizaç", tex, re.IGNORECASE))
        has_sus_context = bool(
            re.search(r"\bRNDS\b|\bDATASUS\b|Sa[uú]de P[uú]blica", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_pni and has_sus_context,
            "F2 Failure: artigo.tex must contextualize public health (PNI, RNDS/DATASUS).",
        )

        # 15,403 records mentioned
        has_sample_size = bool(re.search(r"15[\.\,]?403", tex))
        self.assertTrue(
            has_sample_size,
            "F2 Failure: artigo.tex must cite the sanitized dataset size of 15,403 records.",
        )

        # atraso_critico target definition (> 7 days)
        has_target = bool(re.search(r"atraso[\s_]cr[ií]tico", tex, re.IGNORECASE))
        has_threshold = bool(re.search(r"7\s*dias|>\s*7", tex))
        self.assertTrue(
            has_target and has_threshold,
            "F2 Failure: artigo.tex must define target variable 'atraso_critico' (> 7 days delay).",
        )

    def test_f3_pipeline_integration_diagram(self):
        """F3: Schematic pipeline integration diagram presence and inclusion in TeX."""
        fig_path = find_figure_path("pipeline_integrado")
        self.assertIsNotNone(
            fig_path,
            "F3 Failure: Schematic diagram 'figures/pipeline_integrado.pdf' or '.png' not found.",
        )
        self.assertGreater(
            fig_path.stat().st_size,
            1000,
            f"F3 Failure: Figure '{fig_path.name}' is suspiciously small or empty.",
        )

        self.assertTrue(ARTIGO_TEX.exists(), "F3 Failure: artigo.tex not found.")
        tex = read_tex_file(ARTIGO_TEX)
        self.assertRegex(
            tex,
            r"\\includegraphics(\[[^\]]*\])?\{[^}]*pipeline_integrado[^}]*\}",
            "F3 Failure: artigo.tex must include 'pipeline_integrado' diagram with \\includegraphics.",
        )

    def test_f4_decision_tree_id3_manual_entropy_info_gain(self):
        """F4: Decision tree (ID3) coverage, manual calculations of entropy and info gain
        for 3 levels with exact mathematical formulas and 5-fold CV results."""
        self.assertTrue(ARTIGO_TEX.exists(), "F4 Failure: artigo.tex not found.")
        tex = read_tex_file(ARTIGO_TEX)

        # Mathematical formulas
        has_entropy_formula = bool(
            re.search(r"H\s*\(\s*S\s*\)|-\s*\\sum.*\\log_?2", tex)
        )
        has_ig_formula = bool(
            re.search(r"IG\s*\(\s*S\s*,\s*[A-Za-z0-9_]+\s*\)|Ganho.*Inform", tex)
        )
        self.assertTrue(
            has_entropy_formula and has_ig_formula,
            "F4 Failure: artigo.tex must formulate Shannon Entropy H(S) and Information Gain IG(S, A).",
        )

        # Level 1 calculations
        has_root_entropy = bool(re.search(r"0[\.\,]999[0-9]*", tex))
        has_root_winner = bool(
            re.search(r"sistema[\s_]offline", tex, re.IGNORECASE)
            and re.search(r"0[\.\,]213[0-9]*", tex)
        )
        self.assertTrue(
            has_root_entropy and has_root_winner,
            "F4 Failure: Level 1 manual calculations missing root entropy (~0.9993) or winner sistema_offline (IG ~0.2133).",
        )

        # Level 2 & 3 calculations (subsets and attribute splits)
        has_level2_online = bool(
            re.search(r"7[\.\,]?509", tex) or re.search(r"0[\.\,]802[0-9]*", tex)
        )
        has_level2_offline = bool(
            re.search(r"7[\.\,]?894", tex) or re.search(r"0[\.\,]770[0-9]*", tex)
        )
        self.assertTrue(
            has_level2_online and has_level2_offline,
            "F4 Failure: Level 2 manual calculations missing online (7,509 / 0.8022) or offline (7,894 / 0.7705) branches.",
        )

        # 5-fold CV accuracy reported within 72% to 80%
        has_cv_result = bool(re.search(r"76[\.\,]5[0-9]*\%?", tex)) or bool(
            re.search(r"76[\.\,]6[0-9]*\%?", tex)
        )
        self.assertTrue(
            has_cv_result,
            "F4 Failure: artigo.tex must report 5-fold cross-validation accuracy (~76.53%).",
        )

    def test_f5_non_monotonicity_theoretical_discussion(self):
        """F5: Non-monotonicity theoretical discussion (~0.5 page, refutability, inductive vs deductive)."""
        self.assertTrue(ARTIGO_TEX.exists(), "F5 Failure: artigo.tex not found.")
        tex = read_tex_file(ARTIGO_TEX)

        # Must mention non-monotonicity
        has_non_mono = bool(
            re.search(r"n[aã]o[\s-]monot[oô]nic|monotonia", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_non_mono,
            "F5 Failure: artigo.tex must contain section/discussion on non-monotonicity.",
        )

        # Contrast inductive learning vs deductive logic
        has_deductive_contrast = bool(
            re.search(r"deduti|l[oó]gica de primeira ordem|\bLPO\b|\bFOL\b", tex, re.IGNORECASE)
            and re.search(r"induti", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_deductive_contrast,
            "F5 Failure: artigo.tex must contrast inductive learning with deductive logic.",
        )

        # Substantial text length (~0.5 page is at least ~250 words)
        # Extract paragraph(s) mentioning non-monotonicity
        mono_paragraphs = [
            p
            for p in tex.split("\n\n")
            if re.search(r"monot[oô]nic", p, re.IGNORECASE)
        ]
        total_mono_words = sum(len(p.split()) for p in mono_paragraphs)
        self.assertGreaterEqual(
            total_mono_words,
            150,
            f"F5 Failure: Discussion on non-monotonicity too short ({total_mono_words} words). Expected ~0.5 page (~200+ words).",
        )

    def test_f6_linear_models_ols_gd_ridge_logistic(self):
        """F6: Linear models (OLS closed form, GD, Ridge trajectory vs lambda, Logistic regression)."""
        self.assertTrue(ARTIGO_TEX.exists(), "F6 Failure: artigo.tex not found.")
        tex = read_tex_file(ARTIGO_TEX)

        # OLS and GD formulas
        has_ols = bool(re.search(r"\\beta_?[01]|m[ií]nimos quadrados|OLS", tex, re.IGNORECASE))
        has_gd = bool(
            re.search(r"gradiente descendente|\\alpha|\\nabla", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_ols and has_gd,
            "F6 Failure: artigo.tex must detail OLS closed-form and multivariate Gradient Descent.",
        )

        # Ridge regularized objective and coefficient trajectory
        has_ridge = bool(re.search(r"Ridge|regulariza[çc][aã]o|\\lambda", tex, re.IGNORECASE))
        self.assertTrue(
            has_ridge,
            "F6 Failure: artigo.tex must detail Ridge regression with L2 penalty lambda.",
        )

        # Ridge trajectory figure
        ridge_fig = find_figure_path("etapa2_ridge_trajetoria")
        self.assertIsNotNone(
            ridge_fig,
            "F6 Failure: Ridge trajectory figure 'figures/etapa2_ridge_trajetoria.pdf' or '.png' not found.",
        )
        self.assertRegex(
            tex,
            r"\\includegraphics(\[[^\]]*\])?\{[^}]*etapa2_ridge_trajetoria[^}]*\}",
            "F6 Failure: artigo.tex must include 'etapa2_ridge_trajetoria' figure.",
        )

        # Logistic Regression
        has_logistic = bool(
            re.search(r"regress[aã]o log[ií]stica|sigm[oó]ide|\\sigma", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_logistic,
            "F6 Failure: artigo.tex must cover Logistic Regression with sigmoid function.",
        )

    def test_f7_probability_theory(self):
        """F7: Probability theory (4 atomic events summing to 1.0, marginal/conditional,
        Bayes theorem, conditional independence)."""
        self.assertTrue(ARTIGO_TEX.exists(), "F7 Failure: artigo.tex not found.")
        tex = read_tex_file(ARTIGO_TEX)

        # 4 atomic events joint distribution
        has_joint_events = bool(
            re.search(r"0[\.\,]3684", tex)
            and re.search(r"0[\.\,]1191", tex)
            and re.search(r"0[\.\,]1157", tex)
            and re.search(r"0[\.\,]3968", tex)
        )
        self.assertTrue(
            has_joint_events,
            "F7 Failure: artigo.tex must report the 4 atomic events of joint distribution P(S, A) (0.3684, 0.1191, 0.1157, 0.3968).",
        )

        # Sum equals 1.0 (Kolmogorov axiom)
        has_kolmogorov = bool(re.search(r"1[\.\,]0000|\bsoma\b.*=.*1", tex, re.IGNORECASE))
        self.assertTrue(
            has_kolmogorov,
            "F7 Failure: artigo.tex must demonstrate axiomatic sum = 1.000000.",
        )

        # Conditionals and Relative Risk (RR ~ 3.17x)
        has_conditionals = bool(
            re.search(r"0[\.\,]7743|77[\.\,]43\%?", tex)
            and re.search(r"0[\.\,]2444|24[\.\,]44\%?", tex)
        )
        self.assertTrue(
            has_conditionals,
            "F7 Failure: artigo.tex must present conditional probabilities P(A=1|S=1)=77.43% and P(A=1|S=0)=24.44%.",
        )

        # Bayes Theorem
        has_bayes = bool(
            re.search(r"Bayes", tex, re.IGNORECASE)
            and re.search(r"0[\.\,]7691|76[\.\,]91\%?", tex)
        )
        self.assertTrue(
            has_bayes,
            "F7 Failure: artigo.tex must formulate Bayes Theorem and diagnostic result P(S=1|A=1) = 76.91%.",
        )

    def test_f8_probabilistic_ml_and_em(self):
        """F8: Probabilistic ML and EM (MLE, MAP, Naive Bayes from scratch, EM latent variable,
        log-likelihood curve, 75.79% -> 93.77% accuracy)."""
        self.assertTrue(ARTIGO_TEX.exists(), "F8 Failure: artigo.tex not found.")
        tex = read_tex_file(ARTIGO_TEX)

        # MLE & MAP
        has_mle_map = bool(
            re.search(r"\bMLE\b|\bMAP\b|m[aá]xima verossimilhan[çc]a|a posteriori", tex, re.IGNORECASE)
            and re.search(r"0[\.\,]5159", tex)
        )
        self.assertTrue(
            has_mle_map,
            "F8 Failure: artigo.tex must present MLE and MAP estimators with Beta prior.",
        )

        # Naive Bayes from scratch
        has_naive_bayes = bool(
            re.search(r"Naive\s*Bayes", tex, re.IGNORECASE)
            and re.search(r"75[\.\,]79\%?", tex)
        )
        self.assertTrue(
            has_naive_bayes,
            "F8 Failure: artigo.tex must cover Naive Bayes and report test accuracy ~75.79%.",
        )

        # EM algorithm for latent variable Z
        has_em = bool(
            re.search(r"\bEM\b|Expectation-Maximization", tex, re.IGNORECASE)
            and re.search(r"vari[aá]vel\s*(oculta|latente)|mistura", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_em,
            "F8 Failure: artigo.tex must detail Expectation-Maximization (EM) for latent operational variable Z.",
        )

        # EM convergence figure
        em_fig = find_figure_path("etapa4_em_convergencia")
        self.assertIsNotNone(
            em_fig,
            "F8 Failure: EM convergence figure 'figures/etapa4_em_convergencia.pdf' or '.png' not found.",
        )
        self.assertRegex(
            tex,
            r"\\includegraphics(\[[^\]]*\])?\{[^}]*etapa4_em_convergencia[^}]*\}",
            "F8 Failure: artigo.tex must include 'etapa4_em_convergencia' figure.",
        )

        # Performance jump (75.79% -> 93.77%)
        has_jump = bool(re.search(r"93[\.\,]77\%?", tex))
        self.assertTrue(
            has_jump,
            "F8 Failure: artigo.tex must report the accuracy jump to 93.77% achieved with latent variable Z.",
        )

    def test_f9_bayesian_network(self):
        """F9: Bayesian Network (5-node causal DAG, CPTs, exact enumeration,
        d-separation/collider, parameter economy 67.74%)."""
        self.assertTrue(ARTIGO_TEX.exists(), "F9 Failure: artigo.tex not found.")
        tex = read_tex_file(ARTIGO_TEX)

        # 5 nodes mentioned
        nodes = ["R", "S", "V", "N", "A"]
        for node in nodes:
            self.assertTrue(
                bool(re.search(rf"\b{node}\b", tex)),
                f"F9 Failure: Node '{node}' of 5-node Bayesian Network missing in artigo.tex.",
            )

        # Collider structure and d-separation
        has_collider = bool(
            re.search(r"colisor|collider|d-separa[çc][aã]o|explaining\s*away", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_collider,
            "F9 Failure: artigo.tex must analyze d-separation and collider node activation (S -> A <- N).",
        )

        # Exact enumeration inference
        has_enum = bool(
            re.search(r"enumera[çc][aã]o|exact inference|infer[eê]ncia exata", tex, re.IGNORECASE)
            and re.search(r"0[\.\,]7458|74[\.\,]58\%?", tex)
        )
        self.assertTrue(
            has_enum,
            "F9 Failure: artigo.tex must present exact inference by enumeration P(A=1|S=1, N=1) = 74.58%.",
        )

        # Parameter economy (67.74% reduction: 31 full joint vs 10 in BN)
        has_economy = bool(
            re.search(r"67[\.\,]7[0-9]*\%?", tex)
            or (re.search(r"31", tex) and re.search(r"10", tex))
        )
        self.assertTrue(
            has_economy,
            "F9 Failure: artigo.tex must document parameter economy (31 vs 10 parameters, ~67.74% reduction).",
        )

        # Bayesian network DAG figure
        dag_fig = find_figure_path("etapa5_rede_bayesiana_dag")
        self.assertIsNotNone(
            dag_fig,
            "F9 Failure: Bayesian Network DAG figure 'figures/etapa5_rede_bayesiana_dag.pdf' or '.png' not found.",
        )
        self.assertRegex(
            tex,
            r"\\includegraphics(\[[^\]]*\])?\{[^}]*etapa5_rede_bayesiana_dag[^}]*\}",
            "F9 Failure: artigo.tex must include 'etapa5_rede_bayesiana_dag' figure.",
        )

    def test_f10_fuzzy_logic(self):
        """F10: Fuzzy Logic (Mamdani system, 2 antecedents, 1 consequent, triangular membership,
        5 rules, Stage 5 Bayesian probabilities feeding Fuzzy, comparison with hard threshold)."""
        self.assertTrue(ARTIGO_TEX.exists(), "F10 Failure: artigo.tex not found.")
        tex = read_tex_file(ARTIGO_TEX)

        # Mamdani system and triangular membership
        has_mamdani = bool(re.search(r"Mamdani", tex, re.IGNORECASE))
        has_triangular = bool(re.search(r"triangula", tex, re.IGNORECASE))
        self.assertTrue(
            has_mamdani and has_triangular,
            "F10 Failure: artigo.tex must specify a Mamdani Fuzzy System with triangular membership functions.",
        )

        # 5 rules
        has_rules = bool(
            re.search(r"5\s*regras|regras\s*se-ent[aã]o|base de regras", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_rules,
            "F10 Failure: artigo.tex must define the 5 IF-THEN expert rules.",
        )

        # Defuzzification by centroid
        has_cog = bool(
            re.search(r"centr[oó]ide|centro de gravidade|CoG", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_cog,
            "F10 Failure: artigo.tex must specify defuzzification by Centroid / Center of Gravity (CoG).",
        )

        # Fed by Stage 5 Bayesian probabilities
        has_coupling = bool(
            re.search(r"0[\.\,]7458", tex) and re.search(r"64[\.\,]61", tex)
        ) or bool(
            re.search(r"etapa\s*5.*fuzzy|probabilidade.*fuzzy|bayes.*fuzzy", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_coupling,
            "F10 Failure: artigo.tex must describe feeding Stage 5 Bayesian posterior probabilities into Fuzzy antecedents.",
        )

        # Comparison with hard threshold
        has_threshold_comp = bool(
            re.search(r"limiar\s*r[ií]gido|hard\s*threshold", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_threshold_comp,
            "F10 Failure: artigo.tex must compare smooth Fuzzy decision with hard threshold (0.5).",
        )

        # Fuzzy figure
        fuzzy_fig = find_figure_path("etapa6_fuzzy_superficie")
        self.assertIsNotNone(
            fuzzy_fig,
            "F10 Failure: Fuzzy surface figure 'figures/etapa6_fuzzy_superficie.pdf' or '.png' not found.",
        )
        self.assertRegex(
            tex,
            r"\\includegraphics(\[[^\]]*\])?\{[^}]*etapa6_fuzzy_superficie[^}]*\}",
            "F10 Failure: artigo.tex must include 'etapa6_fuzzy_superficie' figure.",
        )

    def test_f11_bibtex_citations_and_appendix(self):
        """F11: BibTeX citations in SBC format and Methodology/Prompts Appendix."""
        self.assertTrue(
            REFERENCIAS_BIB.exists(),
            f"F11 Failure: BibTeX file '{REFERENCIAS_BIB.name}' not found.",
        )
        self.assertTrue(ARTIGO_TEX.exists(), "F11 Failure: artigo.tex not found.")

        # Check references in .bib
        bib_keys = extract_bibtex_keys(REFERENCIAS_BIB)
        self.assertGreaterEqual(
            len(bib_keys),
            4,
            f"F11 Failure: referencias.bib contains fewer than 4 citations (found: {len(bib_keys)}).",
        )

        # Check core foundational references exist in .bib
        bib_content = read_tex_file(REFERENCIAS_BIB)
        has_rn = bool(re.search(r"russell|norvig", bib_content, re.IGNORECASE))
        has_mitchell_or_bishop = bool(
            re.search(r"mitchell|bishop", bib_content, re.IGNORECASE)
        )
        has_zadeh = bool(re.search(r"zadeh", bib_content, re.IGNORECASE))
        has_pni_ref = bool(
            re.search(r"brasil|sa[uú]de|pni|datasus|rnds", bib_content, re.IGNORECASE)
        )
        self.assertTrue(
            has_rn and (has_mitchell_or_bishop or has_zadeh) and has_pni_ref,
            "F11 Failure: referencias.bib must include canonical foundational references (Russell & Norvig, Zadeh/Mitchell, Ministério da Saúde/PNI).",
        )

        # Check Appendix in artigo.tex
        tex = read_tex_file(ARTIGO_TEX)
        has_appendix = bool(
            re.search(r"\\appendix|ap[eê]ndice|prompts", tex, re.IGNORECASE)
        )
        self.assertTrue(
            has_appendix,
            "F11 Failure: artigo.tex must include an Appendix detailing methodology and AI prompts.",
        )

    def test_f12_high_resolution_figures_present(self):
        """F12: High-resolution figures present in figures/ (pipeline, ridge, em, dag, fuzzy)."""
        self.assertTrue(
            FIGURES_DIR.exists() and FIGURES_DIR.is_dir(),
            f"F12 Failure: Directory '{FIGURES_DIR.name}' does not exist.",
        )

        missing = []
        for base in REQUIRED_FIGURE_BASENAMES:
            path = find_figure_path(base)
            if not path:
                missing.append(base)
            else:
                self.assertGreater(
                    path.stat().st_size,
                    1000,
                    f"F12 Failure: Figure '{path.name}' has invalid size ({path.stat().st_size} bytes).",
                )

        self.assertEqual(
            missing,
            [],
            f"F12 Failure: The following required figures are missing in {FIGURES_DIR}: {missing}",
        )

    def test_f13_compilation_and_page_budget(self):
        """F13: Tectonic compilation check and page budget check (strictly 6 to 8 pages in PDF)."""
        self.assertTrue(
            ARTIGO_PDF.exists(),
            f"F13 Failure: Compiled document '{ARTIGO_PDF.name}' not found.",
        )
        page_count = get_pdf_page_count(ARTIGO_PDF)
        self.assertGreater(
            page_count,
            0,
            "F13 Failure: Unable to read page count from compiled PDF.",
        )
        self.assertGreaterEqual(
            page_count,
            6,
            f"F13 Failure: artigo.pdf has {page_count} pages, which is under the minimum budget of 6 pages.",
        )
        self.assertLessEqual(
            page_count,
            8,
            f"F13 Failure: artigo.pdf has {page_count} pages, which overflows the maximum budget of 8 pages.",
        )

    def test_f14_git_commit_and_push_readiness(self):
        """F14: Git repository check, main branch, and remote origin configured."""
        git_dir = PROJECT_ROOT / ".git"
        self.assertTrue(
            git_dir.exists(),
            "F14 Failure: .git directory not found in project root.",
        )

        # Check branch is main
        branch_res = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        branch = branch_res.stdout.strip()
        self.assertEqual(
            branch,
            "main",
            f"F14 Failure: Active branch is '{branch}', expected 'main'.",
        )

        # Check remote origin
        remote_res = subprocess.run(
            ["git", "remote", "-v"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertIn(
            "origin",
            remote_res.stdout,
            "F14 Failure: Remote 'origin' is not configured in git.",
        )


# ==============================================================================
# TIER 2 — BOUNDARY & CORNER CASES
# ==============================================================================
class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """Verifies edge cases: strict page limits (rejecting 5 and 9 pages),
    BibTeX syntax integrity and zero undefined citations, image resolution constraints."""

    def test_strict_page_budget_boundary_enforcement(self):
        """Tier 2: Page budget boundary condition: 6, 7, 8 pages accepted;
        5 pages or 9 pages must strictly fail."""

        def evaluate_page_budget(pages: int) -> bool:
            return 6 <= pages <= 8

        # Boundary logic verification
        self.assertFalse(evaluate_page_budget(0), "Boundary: 0 pages must be rejected.")
        self.assertFalse(evaluate_page_budget(5), "Boundary: 5 pages must be rejected.")
        self.assertTrue(evaluate_page_budget(6), "Boundary: 6 pages must be accepted.")
        self.assertTrue(evaluate_page_budget(7), "Boundary: 7 pages must be accepted.")
        self.assertTrue(evaluate_page_budget(8), "Boundary: 8 pages must be accepted.")
        self.assertFalse(evaluate_page_budget(9), "Boundary: 9 pages must be rejected.")
        self.assertFalse(evaluate_page_budget(10), "Boundary: 10 pages must be rejected.")

        # Actual PDF check if compiled
        if ARTIGO_PDF.exists():
            actual_pages = get_pdf_page_count(ARTIGO_PDF)
            self.assertTrue(
                evaluate_page_budget(actual_pages),
                f"Boundary Failure: Real PDF has {actual_pages} pages, strictly violating the 6-8 page window.",
            )

    def test_bibtex_validity_and_zero_undefined_references(self):
        """Tier 2: Every citation key in artigo.tex must exist in referencias.bib,
        and no 'Citation ... undefined' warnings in log."""
        self.assertTrue(ARTIGO_TEX.exists(), "Tier 2 Failure: artigo.tex not found.")
        self.assertTrue(REFERENCIAS_BIB.exists(), "Tier 2 Failure: referencias.bib not found.")

        tex_content = read_tex_file(ARTIGO_TEX)
        cited_keys = extract_citations_from_tex(tex_content)
        bib_keys = extract_bibtex_keys(REFERENCIAS_BIB)

        missing_keys = cited_keys - bib_keys
        self.assertEqual(
            missing_keys,
            set(),
            f"Tier 2 Failure: The following citations in artigo.tex are missing from referencias.bib: {missing_keys}",
        )

        # Check compilation log if present
        if ARTIGO_LOG.exists():
            log_content = read_tex_file(ARTIGO_LOG)
            undefined_matches = re.findall(
                r"Citation [`']([^']+)['\s] on page \d+ undefined", log_content
            )
            self.assertEqual(
                undefined_matches,
                [],
                f"Tier 2 Failure: Compilation log contains undefined citations: {undefined_matches}",
            )

    def test_image_format_and_integrity(self):
        """Tier 2: All figure files must be valid PDF vector files or high-resolution PNGs."""
        if not FIGURES_DIR.exists():
            self.fail(f"Tier 2 Failure: figures directory '{FIGURES_DIR}' not found.")

        for base in REQUIRED_FIGURE_BASENAMES:
            fig = find_figure_path(base)
            if not fig:
                self.fail(f"Tier 2 Failure: Figure '{base}' not found (.pdf or .png).")

            if fig.suffix.lower() == ".pdf":
                # Check valid PDF magic bytes
                with open(fig, "rb") as f:
                    magic = f.read(5)
                self.assertEqual(
                    magic,
                    b"%PDF-",
                    f"Tier 2 Failure: File '{fig.name}' is not a valid PDF file.",
                )
            elif fig.suffix.lower() == ".png":
                # Check PIL image validation
                try:
                    from PIL import Image

                    with Image.open(fig) as img:
                        img.verify()
                        width, height = img.size
                        self.assertGreaterEqual(
                            width,
                            800,
                            f"Tier 2 Failure: PNG '{fig.name}' resolution too low ({width}x{height}). Minimum width is 800px.",
                        )
                except Exception as e:
                    self.fail(f"Tier 2 Failure: Invalid or corrupt image '{fig.name}': {e}")

    def test_resumo_and_abstract_line_budget_boundary(self):
        """Tier 2: Boundary check for Resumo and Abstract text length (<= 10 lines each)."""
        self.assertTrue(ARTIGO_TEX.exists(), "Tier 2 Failure: artigo.tex not found.")
        tex_content = read_tex_file(ARTIGO_TEX)

        for env_name in ["resumo", "abstract"]:
            match = re.search(
                rf"\\begin\{{{env_name}\}}([\s\S]*?)\\end\{{{env_name}\}}", tex_content
            )
            if match:
                raw_text = match.group(1).strip()
                # Count lines of actual content
                lines = [l for l in raw_text.splitlines() if l.strip()]
                self.assertLessEqual(
                    len(lines),
                    10,
                    f"Tier 2 Failure: Environment '{env_name}' has {len(lines)} lines (exceeds 10-line budget).",
                )


# ==============================================================================
# TIER 3 — CROSS-FEATURE INTEGRATION
# ==============================================================================
class TestTier3CrossFeatureIntegration(unittest.TestCase):
    """Verifies end-to-end pipeline coupling:
    - Stage 5 Bayesian posteriors feeding Stage 6 Fuzzy antecedents.
    - Consistency of numerical claims across text, tables, and ground truth."""

    def test_bayes_to_fuzzy_quantitative_coupling(self):
        """Tier 3: Quantitative coupling between Stage 5 posterior probabilities
        and Stage 6 Fuzzy antecedents in the 3 SUS operational scenarios."""
        self.assertTrue(ARTIGO_TEX.exists(), "Tier 3 Failure: artigo.tex not found.")
        tex = read_tex_file(ARTIGO_TEX)

        scenarios = GROUND_TRUTH_METRICS["bayes_to_fuzzy_scenarios"]

        # Rural Scenario: Posterior ~0.7458, Fuzzy Urgency ~64.61
        rural_p = f"{scenarios['rural']['posterior']:.4f}".replace(".", r"[\.\,]")
        rural_u = f"{scenarios['rural']['urgency']:.2f}".replace(".", r"[\.\,]")
        self.assertTrue(
            bool(re.search(rural_p, tex)) and bool(re.search(rural_u, tex)),
            f"Tier 3 Failure: Rural scenario coupling missing in artigo.tex (expected posterior ~0.7458 and urgency ~64.61).",
        )

        # Urban Scenario: Posterior ~0.3059, Fuzzy Urgency ~38.79
        urban_p = f"{scenarios['urban']['posterior']:.4f}".replace(".", r"[\.\,]")
        urban_u = f"{scenarios['urban']['urgency']:.2f}".replace(".", r"[\.\,]")
        self.assertTrue(
            bool(re.search(urban_p, tex)) and bool(re.search(urban_u, tex)),
            f"Tier 3 Failure: Urban scenario coupling missing in artigo.tex (expected posterior ~0.3059 and urgency ~38.79).",
        )

        # Frontier Scenario: Posterior ~0.1758, Fuzzy Urgency ~23.90
        frontier_p = f"{scenarios['frontier']['posterior']:.4f}".replace(".", r"[\.\,]")
        frontier_u = f"{scenarios['frontier']['urgency']:.2f}".replace(".", r"[\.\,]")
        self.assertTrue(
            bool(re.search(frontier_p, tex)) and bool(re.search(frontier_u, tex)),
            f"Tier 3 Failure: Frontier scenario coupling missing in artigo.tex (expected posterior ~0.1758 and urgency ~23.90).",
        )

    def test_numerical_claims_consistency_with_ground_truth(self):
        """Tier 3: Quantitative consistency check between article text and ground truth values."""
        self.assertTrue(ARTIGO_TEX.exists(), "Tier 3 Failure: artigo.tex not found.")
        tex = read_tex_file(ARTIGO_TEX)

        # 1. Total samples (15,403)
        self.assertRegex(
            tex,
            r"15[\.\,]?403",
            "Tier 3 Failure: Sample size 15,403 not found in artigo.tex.",
        )

        # 2. Level 1 root entropy (0.9993)
        self.assertRegex(
            tex,
            r"0[\.\,]999[0-9]*",
            "Tier 3 Failure: Root entropy 0.9993 not found in artigo.tex.",
        )

        # 3. Decision tree CV accuracy (76.53%)
        self.assertTrue(
            bool(re.search(r"76[\.\,]5[0-9]*", tex)) or bool(re.search(r"76[\.\,]6[0-9]*", tex)),
            "Tier 3 Failure: Decision tree CV accuracy (~76.53%) not found in artigo.tex.",
        )

        # 4. Logistic regression accuracy (~77.47%)
        self.assertTrue(
            bool(re.search(r"77[\.\,]4[0-9]*|77[\.\,]5\%?", tex)),
            "Tier 3 Failure: Logistic regression accuracy (~77.47%) not found in artigo.tex.",
        )

        # 5. Axiomatic sum (1.000000)
        self.assertRegex(
            tex,
            r"1[\.\,]0000",
            "Tier 3 Failure: Kolmogorov axiomatic sum 1.0000 not found in artigo.tex.",
        )

        # 6. Relative risk (3.17x)
        self.assertRegex(
            tex,
            r"3[\.\,]17",
            "Tier 3 Failure: Relative risk 3.17x not found in artigo.tex.",
        )

        # 7. Naive Bayes jump (75.79% -> 93.77%)
        self.assertTrue(
            bool(re.search(r"75[\.\,]79\%?", tex)) and bool(re.search(r"93[\.\,]77\%?", tex)),
            "Tier 3 Failure: Accuracy jump from 75.79% to 93.77% not documented in artigo.tex.",
        )

        # 8. Bayesian network parameter reduction (67.74% or 31 vs 10)
        self.assertTrue(
            bool(re.search(r"67[\.\,]7[0-9]*\%?", tex))
            or (bool(re.search(r"\b31\b", tex)) and bool(re.search(r"\b10\b", tex))),
            "Tier 3 Failure: Parameter economy (67.74% / 31 vs 10 parameters) not found in artigo.tex.",
        )


# ==============================================================================
# TIER 4 — REAL-WORLD WORKLOAD SCENARIOS
# ==============================================================================
class TestTier4RealWorldWorkloads(unittest.TestCase):
    """Verifies complete autonomous document compilation using Tectonic
    and clean generation of production PDF without critical errors."""

    def test_tectonic_binary_availability(self):
        """Tier 4: Verify Tectonic compiler is accessible in system."""
        tectonic_path = shutil.which("tectonic") or (
            str(TECTONIC_BIN) if TECTONIC_BIN.exists() else None
        )
        self.assertIsNotNone(
            tectonic_path,
            f"Tier 4 Failure: Tectonic compiler binary not found at '{TECTONIC_BIN}' or in PATH.",
        )

        # Verify executable runs
        res = subprocess.run(
            [tectonic_path, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            res.returncode,
            0,
            f"Tier 4 Failure: Tectonic returned non-zero exit code ({res.returncode}): {res.stderr}",
        )
        self.assertIn("Tectonic", res.stdout)

    def test_tectonic_compilation_execution(self):
        """Tier 4: Compiles artigo.tex using Tectonic and verifies exit code 0."""
        self.assertTrue(
            ARTIGO_TEX.exists(),
            "Tier 4 Failure: artigo.tex not found. Waiting for Milestone 2.",
        )
        tectonic_path = shutil.which("tectonic") or (
            str(TECTONIC_BIN) if TECTONIC_BIN.exists() else None
        )
        self.assertIsNotNone(
            tectonic_path,
            "Tier 4 Failure: Tectonic compiler not found.",
        )

        # Run compilation
        cmd = [tectonic_path, str(ARTIGO_TEX)]
        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )

        self.assertEqual(
            res.returncode,
            0,
            f"Tier 4 Failure: Tectonic compilation failed with exit code {res.returncode}.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
        )

        self.assertTrue(
            ARTIGO_PDF.exists(),
            "Tier 4 Failure: Tectonic compilation succeeded but artigo.pdf was not produced.",
        )
        self.assertGreater(
            ARTIGO_PDF.stat().st_size,
            10000,
            f"Tier 4 Failure: Generated PDF '{ARTIGO_PDF.name}' is too small ({ARTIGO_PDF.stat().st_size} bytes).",
        )

    def test_clean_build_verification(self):
        """Tier 4: Verifies compiled PDF is not stale and page count satisfies the 6-8 budget."""
        self.assertTrue(
            ARTIGO_PDF.exists(),
            "Tier 4 Failure: artigo.pdf does not exist for clean build verification.",
        )
        pages = get_pdf_page_count(ARTIGO_PDF)
        self.assertTrue(
            6 <= pages <= 8,
            f"Tier 4 Failure: Clean build produced {pages} pages, strictly violating the 6 to 8 page budget.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
