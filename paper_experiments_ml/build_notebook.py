"""Build the single, Colab-runnable manuscript experiment notebook.

The notebook is generated from the canonical experiment scripts and contracts.
Every code cell is preceded by mathematical/algorithmic Markdown. A complete
data atlas after every experiment plots every CSV cell and every non-row JSON
or raw-JSON leaf, alongside a relationship-focused interactive view.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from publication import atomic_write_text, index_rows

ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "PhotoGraphiQML_Manuscript_Experiments.ipynb"
CONTRACTS_PATH = ROOT / "EXPERIMENT_CONTRACTS.json"
SYS_PATH_LINE_RE = re.compile(
    r"^sys\.path\.insert\(0, str\(Path\(__file__\)\.resolve\(\)\.parents\[1\]\)\)\n",
    re.MULTILINE,
)


def _contracts() -> dict[str, dict]:
    payload = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    return {item["experiment_id"]: item for item in payload["contracts"]}


def _module_documentation(source: str, fallback: str) -> tuple[str, str]:
    document = ast.get_docstring(ast.parse(source), clean=False) or fallback
    lines = document.strip().splitlines()
    return (lines[0].strip() if lines else fallback, "\n".join(lines[1:]).strip())


def _metric_mathematics(metric: str) -> str:
    lowered = metric.casefold()
    if "frobenius" in lowered or "density-matrix error" in lowered:
        return r"$\lVert A-B\rVert_F=\sqrt{\sum_{ij}|A_{ij}-B_{ij}|^2}$."
    if "tv" in lowered or "total-variation" in lowered:
        return r"$D_{\mathrm{TV}}(p,q)=\tfrac12\sum_x|p(x)-q(x)|$."
    if "rmse" in lowered:
        return r"$\mathrm{RMSE}=\sqrt{n^{-1}\sum_i(\hat y_i-y_i)^2}$."
    if "mae" in lowered:
        return r"$\mathrm{MAE}=n^{-1}\sum_i|\hat y_i-y_i|$."
    if "r2" in lowered or "r^2" in lowered:
        return r"$R^2=1-\sum_i(y_i-\hat y_i)^2/\sum_i(y_i-\bar y)^2$."
    if "accuracy" in lowered:
        return r"$\mathrm{accuracy}=n^{-1}\sum_i\mathbf{1}[\hat y_i=y_i]$."
    if "fidelity" in lowered or "infidelity" in lowered:
        return r"For pure targets, $F=|\langle\psi|\phi\rangle|^2$ and infidelity is $1-F$."
    if "qfi" in lowered or "fisher" in lowered:
        return r"For a pure unitary family, $F_Q=4(\langle H^2\rangle-\langle H\rangle^2)$."
    if "concurrence" in lowered:
        return r"Concurrence follows from the ordered square roots of $\rho\tilde\rho$."
    if "eigenvalue" in lowered or "psd" in lowered or "condition" in lowered:
        return r"PSD requires $\lambda_{\min}(K)\ge-\varepsilon$; $\kappa(K)=\lambda_{\max}/\lambda_{\min}$."
    if "correlation" in lowered:
        return r"$\rho_{XY}=\mathrm{Cov}(X,Y)/(\sigma_X\sigma_Y)$."
    if "probability" in lowered or "branch" in lowered or "Σp" in metric:
        return r"Branch completeness requires $p_b\ge0$ and $\sum_b p_b=1$ within tolerance."
    if "median seconds" in lowered or "peak bytes" in lowered:
        return r"Timing uses the median and $\mathrm{IQR}=Q_{0.75}-Q_{0.25}$."
    if "gradient" in lowered:
        return (
            r"Centered differences $(f(\theta+h)-f(\theta-h))/(2h)$ test parameter-shift gradients."
        )
    if "count" in lowered or "dimension" in lowered:
        return r"The residual is the absolute difference between implemented and predicted counts."
    return r"The declared metric $m$ is checked against the preregistered set $\mathcal A$."


def _experiment_markdown(title: str, body: str, contract: dict) -> str:
    seeds = contract.get("seed_set") or []
    seed_text = ", ".join(map(str, seeds)) if seeds else "deterministic / not applicable"
    return dedent(
        f"""
        ## {title}

        {body}

        ### Mathematical model and declared contract

        **Scientific question.** {contract["scientific_question"]}

        **Hypothesis.** {contract["hypothesis"]}

        **Metric.** {contract["metric_definitions"]} ({contract["metric_units"]}).
        {_metric_mathematics(contract["metric_definitions"])}

        **Oracle/baseline.** {contract["independent_oracle_or_baseline"]}
        (independence class **{contract["oracle_independence_class"]}**).

        **Acceptance rule.** `{contract["exact_acceptance_condition"]}`

        ### Algorithm executed by the next cell

        1. Construct {contract["dataset_or_generated_data_protocol"]}.
        2. Apply preprocessing: {contract["preprocessing"]}.
        3. Evaluate `{contract["implementation_api"]}` and the matched independent oracle.
        4. Compute the metric with {contract["uncertainty_method"]}.
        5. Test the declared condition and atomically save CSV, JSON, metadata, and figures.

        Seeds: **{seed_text}**. Repetitions: **{contract["number_of_repetitions"]}**.
        Hyperparameter protocol: {contract["hyperparameter_selection_protocol"]}

        Supported claim: {contract["supported_claim"]}

        Limitation: {contract["limitations"]}
        """
    ).strip()


def _notebook_source(source: str, experiment_id: str) -> str:
    source = SYS_PATH_LINE_RE.sub("", source)
    source = source.replace(
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))",
        "sys.path.insert(0, str(EXPERIMENT_ROOT))",
    )
    source = source.replace("sys.exit(0)", "raise SystemExit(0)")
    return (
        f"# {experiment_id}: execute the canonical script under the contract above.\n"
        "# Keep declared seeds, tolerances, baselines, and sample sizes unchanged.\n" + source
    )


VISUALIZATION_HELPERS = dedent(
    r'''
    # Complete-data visualization: heatmaps encode every cell, while adaptive
    # companion views reveal trajectories, geometry, distributions, and trade-offs.
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from IPython.display import Image, Markdown, display

    PLOT_TEMPLATE = "plotly_white"
    PLOT_COLORS = ["#24677b", "#c66d27", "#627a36", "#5b4b8a", "#b23a48"]


    def _flatten_leaves(value, prefix="root"):
        """Yield every scalar JSON leaf and its semantic path."""
        if isinstance(value, dict):
            for key, child in value.items():
                yield from _flatten_leaves(child, f"{prefix}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from _flatten_leaves(child, f"{prefix}[{index}]")
        else:
            yield prefix, value


    def _result_paths(experiment_id):
        matches = sorted(
            path
            for path in (EXPERIMENT_ROOT / "results" / "json").glob(f"{experiment_id}_*.json")
            if ".metadata." not in path.name
        )
        if not matches:
            raise FileNotFoundError(f"No JSON result found for {experiment_id}")
        result_path = matches[0]
        stem = result_path.stem
        return (
            result_path,
            EXPERIMENT_ROOT / "results" / "json" / f"{stem}.metadata.json",
            EXPERIMENT_ROOT / "results" / "csv" / f"{stem}.csv",
            EXPERIMENT_ROOT / "figures" / "png" / f"{stem}.png",
        )


    def _numeric_and_categorical(frame):
        """Preserve every cell while separating numeric and categorical encodings."""
        numeric = pd.DataFrame(index=frame.index)
        categorical = pd.DataFrame(index=frame.index)
        for column in frame.columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().sum() == frame[column].notna().sum() and converted.notna().any():
                numeric[column] = converted.astype(float)
            else:
                labels = frame[column].fillna("<missing>").astype(str)
                categorical[column] = pd.factorize(labels, sort=True)[0].astype(float)
        return numeric, categorical


    def _robust_scale(numeric):
        """Make tiny errors and large resource values simultaneously visible."""
        if numeric.empty:
            return numeric
        center = numeric.median(axis=0)
        spread = numeric.quantile(0.75) - numeric.quantile(0.25)
        spread = spread.mask(spread.abs() < np.finfo(float).eps, 1.0)
        return (numeric - center) / spread


    def _show_complete_matrix(frame, title):
        """Plot every numeric and categorical table cell with exact hover values."""
        numeric, categorical = _numeric_and_categorical(frame)
        if not numeric.empty:
            scaled = _robust_scale(numeric).replace([np.inf, -np.inf], np.nan)
            figure = go.Figure(go.Heatmap(
                z=scaled.to_numpy().T,
                x=[str(index) for index in frame.index],
                y=list(numeric.columns),
                customdata=numeric.to_numpy().T,
                colorscale="RdBu", zmid=0, colorbar_title="robust z",
                hovertemplate="row=%{x}<br>field=%{y}<br>raw=%{customdata}<extra></extra>",
            ))
            figure.update_layout(
                title=f"{title}: every numeric cell", template=PLOT_TEMPLATE,
                height=max(380, 24 * len(numeric.columns) + 150),
                xaxis_title="row index", yaxis_title="field",
            )
            figure.show()
        if not categorical.empty:
            figure = go.Figure(go.Heatmap(
                z=categorical.to_numpy().T,
                x=[str(index) for index in frame.index],
                y=list(categorical.columns),
                customdata=frame[categorical.columns].fillna("<missing>").astype(str).to_numpy().T,
                colorscale="Viridis", showscale=False,
                hovertemplate="row=%{x}<br>field=%{y}<br>value=%{customdata}<extra></extra>",
            ))
            figure.update_layout(
                title=f"{title}: every categorical cell", template=PLOT_TEMPLATE,
                height=max(320, 25 * len(categorical.columns) + 140),
                xaxis_title="row index", yaxis_title="field",
            )
            figure.show()
        return numeric


    def _adaptive_scientific_view(numeric, experiment_id):
        """Choose a relationship plot; the complete heatmap still guarantees full coverage."""
        if numeric.empty:
            return
        clean = numeric.replace([np.inf, -np.inf], np.nan).copy()
        clean = clean.fillna(clean.median(numeric_only=True)).fillna(0.0)
        columns = list(clean.columns)
        variant = sum(ord(char) for char in experiment_id) % 6

        if len(columns) >= 4 and variant == 0:
            figure = px.parallel_coordinates(
                _robust_scale(clean).fillna(0.0), dimensions=columns,
                color=np.arange(len(clean)), color_continuous_scale="Turbo",
                title=f"{experiment_id}: multivariate parallel-coordinate trajectories",
            )
        elif len(columns) >= 3 and variant == 1:
            figure = px.scatter_3d(
                clean.reset_index(drop=True), x=columns[0], y=columns[1], z=columns[2],
                color=clean.index.astype(str), title=f"{experiment_id}: three-metric geometry",
            )
            figure.update_traces(marker={"size": 5, "opacity": 0.8})
        elif len(columns) >= 3 and variant == 2:
            figure = px.scatter_matrix(
                clean, dimensions=columns[: min(6, len(columns))],
                color=clean.index.astype(str), title=f"{experiment_id}: pairwise metric geometry",
            )
            figure.update_traces(diagonal_visible=False, showupperhalf=False)
        elif len(columns) >= 3 and variant == 3:
            scaled = _robust_scale(clean).fillna(0.0)
            center = scaled.median(axis=0)
            spread = scaled.quantile(0.75) - scaled.quantile(0.25)
            theta = columns + [columns[0]]
            figure = go.Figure()
            figure.add_trace(go.Scatterpolar(
                r=list(center) + [center.iloc[0]], theta=theta, fill="toself", name="median"
            ))
            figure.add_trace(go.Scatterpolar(
                r=list(spread) + [spread.iloc[0]], theta=theta, fill="toself", name="IQR"
            ))
            figure.update_layout(title=f"{experiment_id}: robust radar profile")
        elif len(columns) >= 2 and variant == 4:
            melted = clean.melt(var_name="metric", value_name="value")
            figure = px.violin(
                melted, x="metric", y="value", color="metric", points="all", box=True,
                title=f"{experiment_id}: complete-observation distributions",
            )
        elif len(columns) >= 2:
            figure = px.line(
                clean.reset_index(), x=columns[0], y=columns[1], markers=True,
                hover_data=["index", *columns[2: min(6, len(columns))]],
                title=f"{experiment_id}: ordered principal-metric relationship",
            )
        else:
            values = np.sort(clean[columns[0]].to_numpy())
            probability = np.arange(1, len(values) + 1) / max(len(values), 1)
            figure = go.Figure(go.Scatter(x=values, y=probability, mode="lines+markers"))
            figure.update_layout(
                title=f"{experiment_id}: empirical cumulative distribution",
                xaxis_title=columns[0], yaxis_title="empirical probability",
            )
        figure.update_layout(template=PLOT_TEMPLATE, coloraxis_showscale=False)
        figure.show()


    def _show_json_leaves(payloads, experiment_id):
        """Plot every non-tabular result, metadata, and raw-data leaf."""
        leaves = []
        for source_name, payload in payloads:
            if isinstance(payload, dict):
                payload = {key: value for key, value in payload.items() if key != "rows"}
            leaves.extend((source_name, path, value) for path, value in _flatten_leaves(payload))
        if not leaves:
            return
        leaf_frame = pd.DataFrame(leaves, columns=["source", "path", "value"])
        numeric_values = pd.to_numeric(leaf_frame["value"], errors="coerce")
        numeric_mask = numeric_values.notna()
        if numeric_mask.any():
            raw = numeric_values[numeric_mask].astype(float)
            transformed = np.sign(raw) * np.log10(1.0 + np.abs(raw))
            figure = px.scatter(
                x=np.arange(len(raw)), y=transformed,
                color=leaf_frame.loc[numeric_mask, "source"],
                hover_name=leaf_frame.loc[numeric_mask, "path"], custom_data=[raw],
                title=f"{experiment_id}: every numeric JSON/metadata leaf",
                labels={"x": "leaf index", "y": "signed log10(1 + |value|)"},
            )
            figure.update_traces(
                marker={"size": 9, "opacity": 0.75},
                hovertemplate="%{hovertext}<br>raw=%{customdata[0]}<extra></extra>",
            )
            figure.update_layout(template=PLOT_TEMPLATE)
            figure.show()
        categorical = leaf_frame.loc[~numeric_mask].copy()
        if not categorical.empty:
            categorical["code"] = pd.factorize(categorical["value"].astype(str), sort=True)[0]
            figure = px.scatter(
                categorical, x=np.arange(len(categorical)), y="code", color="source",
                hover_name="path", hover_data={"value": True, "code": False},
                title=f"{experiment_id}: every categorical JSON/metadata leaf",
                labels={"x": "leaf index", "code": "factorized semantic value"},
            )
            figure.update_traces(marker={"size": 9, "symbol": "diamond", "opacity": 0.75})
            figure.update_layout(template=PLOT_TEMPLATE)
            figure.show()


    def plot_experiment_data(experiment_id):
        """Display the publication figure and complete interactive data atlas."""
        result_path, metadata_path, csv_path, figure_path = _result_paths(experiment_id)
        display(Markdown(f"#### {experiment_id} publication figure and complete data atlas"))
        if figure_path.exists():
            display(Image(filename=str(figure_path), width=1000))

        # The CSV table is shown verbatim, then every one of its cells is plotted.
        frame = pd.read_csv(csv_path)
        display(frame)
        numeric = _show_complete_matrix(frame, f"{experiment_id} row data")
        _adaptive_scientific_view(numeric, experiment_id)

        # Nested result/metadata values and confidence intervals are plotted leaf-for-leaf.
        payloads = [
            ("result", json.loads(result_path.read_text(encoding="utf-8"))),
            ("metadata", json.loads(metadata_path.read_text(encoding="utf-8"))),
        ]
        raw_root = EXPERIMENT_ROOT / "results" / "raw"
        for candidate in sorted(raw_root.glob(f"{experiment_id}*")):
            raw_files = [candidate] if candidate.is_file() else sorted(candidate.rglob("*.json"))
            for raw_path in raw_files:
                try:
                    payloads.append(
                        (f"raw:{raw_path.name}", json.loads(raw_path.read_text(encoding="utf-8")))
                    )
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # Preserve non-JSON text byte-for-byte in the plotted leaf sequence.
                    payloads.append((f"raw-bytes:{raw_path.name}", list(raw_path.read_bytes())))
        _show_json_leaves(payloads, experiment_id)
    '''
).strip()


def _code_cell(source: str, tags: list[str] | None = None):
    cell = nbf.v4.new_code_cell(source)
    if tags:
        cell.metadata["tags"] = tags
    return cell


def _markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def _colab_install_cells():
    explanation = _markdown_cell(
        r"""
        ## Colab bootstrap and dependency installation

        The next cell makes this single file runnable locally or in Google Colab. In Colab it
        clones the unified PhotoGraphiQ repository; locally it uses the repository root that
        contains this suite. It then installs the repository in editable mode with the
        experiment, visualization, validation, and physical-simulation dependencies. The full
        physical and repeated-training suite is intentionally expensive; a high-RAM CPU runtime is
        recommended.
        """
    )
    code = _code_cell(
        dedent(
            r"""
            # Resolve or obtain the repository before importing experiment code.
            import importlib.util
            import os
            import subprocess
            import sys
            from pathlib import Path

            # Colab only: the unified repository that ships photographiq and photographiqml.
            REPOSITORY_URL = "https://github.com/chinmoybiswasdeep/PhotoGraphiQ.git"
            REPOSITORY_BRANCH = "main"
            IN_COLAB = importlib.util.find_spec("google.colab") is not None

            if IN_COLAB:
                repository_root = Path("/content/PhotoGraphiQ")
                if not repository_root.exists():
                    # A shallow branch clone keeps notebook setup reasonably compact.
                    subprocess.run(
                        ["git", "clone", "--depth", "1", "--branch", REPOSITORY_BRANCH,
                         REPOSITORY_URL, str(repository_root)], check=True,
                    )
            else:
                current = Path.cwd().resolve()
                repository_root = current.parent if current.name == "paper_experiments_ml" else current
                if not (repository_root / "pyproject.toml").exists():
                    raise FileNotFoundError("Run locally from the PhotoGraphiQ repository root.")

            os.chdir(repository_root)
            requirements = [
                "-e", ".[ml-experiments,visualization,ml-validation]", "piquasso>=8,<9",
                "pandas>=2", "seaborn>=0.13", "plotly>=5.20", "nbformat>=5",
            ]
            # Install into the active kernel so later cells can import immediately.
            subprocess.run([sys.executable, "-m", "pip", "install", *requirements], check=True)
            print(f"Repository ready at {repository_root}")
            """
        ).strip(),
        ["colab-bootstrap"],
    )
    return [explanation, code]


def _setup_cells():
    explanation = _markdown_cell(
        r"""
        ## Shared execution and visualization setup

        All scripts share deterministic random-number construction, atomic evidence persistence,
        and a common plotting policy. For a table $X$, complete-cell heatmaps use
        $Z_{ij}=(X_{ij}-\operatorname{median}X_{:j})/\operatorname{IQR}(X_{:j})$ while hover text
        retains the raw value. Categorical cells are factorized only for color. The next cell
        defines those transformations and all adaptive relationship plots used below.
        """
    )
    setup_prelude = dedent(
        """
            # Prioritize this checkout over unrelated installed packages with the same name.
            import sys
            from pathlib import Path

            REPOSITORY_ROOT = Path.cwd().resolve()
            EXPERIMENT_ROOT = REPOSITORY_ROOT / "paper_experiments_ml"
            if not (EXPERIMENT_ROOT / "common.py").exists():
                raise FileNotFoundError("paper_experiments_ml/common.py was not found after bootstrap.")
            for path in (REPOSITORY_ROOT / "src", EXPERIMENT_ROOT):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))

            import common
            import metadata

            print(metadata.collect())
            """
    ).strip()
    # Concatenate after dedenting: the helper block contains top-level lines, which would
    # otherwise prevent textwrap.dedent from removing the prelude's indentation.
    setup = _code_cell(
        f"{setup_prelude}\n\n{VISUALIZATION_HELPERS}",
        ["shared-setup", "visualization-atlas"],
    )
    return [explanation, setup]


def _experiment_cells(row: dict, contract: dict):
    path = ROOT / row["script"]
    source = path.read_text(encoding="utf-8")
    title, body = _module_documentation(source, path.stem)
    experiment_id = row["experiment_id"]
    heavy = {"R17", "R22", "R44", "R48", "R52", "R57"}
    return [
        _markdown_cell(_experiment_markdown(title, body, contract)),
        _code_cell(
            _notebook_source(source, experiment_id),
            ["experiment", experiment_id, "heavy" if experiment_id in heavy else "standard"],
        ),
        _markdown_cell(
            f"""
            ### {experiment_id} complete-data visual interpretation

            The next cell displays the publication figure, plots every CSV datum in numeric and
            categorical complete-cell maps, and adds a relationship view selected from parallel
            coordinates, 3-D trajectories, scatter matrices, radar profiles, violin-with-points,
            connected scatter, or ECDF. It then plots every non-row result, confidence interval,
            claim, provenance, metadata, and raw JSON leaf with exact values available on hover.
            """
        ),
        _code_cell(
            f"# Plot every persisted datum for {experiment_id}; hover for exact raw values.\n"
            f'plot_experiment_data("{experiment_id}")\n',
            ["data-complete-plot", experiment_id],
        ),
    ]


def _final_cells():
    return [
        _markdown_cell(
            r"""
            ## Aggregate manuscript tables

            The next cell rebuilds manuscript tables directly from the result schema. It performs
            no refitting and does not reinterpret structural execution as scientific success.
            """
        ),
        _code_cell(
            "# Rebuild CSV and Markdown manuscript tables from the completed results.\n"
            "import generate_tables\n"
            "generate_tables.main()\n",
            ["aggregate-tables"],
        ),
        _markdown_cell(
            r"""
            ## Cross-experiment scientific-outcome topology

            Each experiment becomes one inspectable point in canonical order. Vertical position is
            the scientific outcome, color is oracle-independence class, and symbol is outcome type.
            This preserves all 59 identities and negative results without reducing them to a bar
            count or incorrectly equating structural pass with a positive scientific finding.
            """
        ),
        _code_cell(
            dedent(
                r"""
                # Read one canonical point per experiment and preserve its claims in hover data.
                contracts = json.loads(
                    (EXPERIMENT_ROOT / "EXPERIMENT_CONTRACTS.json").read_text(encoding="utf-8")
                )["contracts"]
                overview = []
                for order, contract in enumerate(contracts):
                    experiment_id = contract["experiment_id"]
                    result_path, _, _, _ = _result_paths(experiment_id)
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    overview.append({
                        "order": order + 1, "experiment": experiment_id,
                        "outcome": result["scientific_outcome"],
                        "structural": result["structural_status"], "oracle": result["oracle_class"],
                        "metric": contract["metric_definitions"],
                        "supported": result["claim_supported"],
                        "not_supported": result["claim_not_supported"],
                    })
                overview = pd.DataFrame(overview)
                outcome_order = {"negative": 0, "inconclusive": 1, "descriptive": 2, "positive": 3}
                overview["outcome_level"] = overview["outcome"].map(outcome_order)
                figure = px.scatter(
                    overview, x="order", y="outcome_level", color="oracle", symbol="outcome",
                    hover_name="experiment", hover_data=["metric", "supported", "not_supported"],
                    title="PhotoGraphiQML: 59-experiment evidence topology",
                    labels={"order": "canonical experiment order", "outcome_level": "scientific outcome"},
                    color_discrete_sequence=PLOT_COLORS,
                )
                figure.update_yaxes(
                    tickmode="array", tickvals=list(outcome_order.values()),
                    ticktext=list(outcome_order.keys()),
                )
                figure.update_traces(marker={"size": 11, "line": {"width": 1, "color": "white"}})
                figure.update_layout(template=PLOT_TEMPLATE, height=520)
                figure.show()
                display(overview)
                """
            ).strip(),
            ["cross-experiment-dashboard"],
        ),
    ]


def _validate_cell_structure(cells) -> None:
    for index, cell in enumerate(cells):
        if cell.cell_type == "code":
            if index == 0 or cells[index - 1].cell_type != "markdown":
                raise ValueError(f"Code cell {index} lacks a preceding Markdown explanation")
            if "#" not in cell.source:
                raise ValueError(f"Code cell {index} lacks inline explanatory comments")


def main() -> None:
    contracts = _contracts()
    rows = index_rows()
    cells = [
        _markdown_cell(
            r"""
            # PhotoGraphiQML: complete Colab manuscript experiment suite

            This single notebook contains all **59** manuscript experiments: R1–R48, R_PERF,
            and R50–R59. Every executable cell has a preceding mathematical/algorithmic Markdown
            explanation and inline comments. Each following interactive atlas represents every
            persisted tabular cell and nested result/metadata/raw-JSON leaf, paired with useful
            relationship plots rather than relying on bar charts and histograms.

            Negative findings remain negative: structural execution is never relabeled as
            scientific success. Executing the complete physical/statistical suite is expensive;
            use a Colab high-RAM CPU runtime and keep the session active.
            """
        )
    ]
    cells.extend(_colab_install_cells())
    cells.extend(_setup_cells())
    for row in rows:
        cells.extend(_experiment_cells(row, contracts[row["experiment_id"]]))
    cells.extend(_final_cells())
    _validate_cell_structure(cells)

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"] = {
        "accelerator": "CPU",
        "colab": {"name": NOTEBOOK_PATH.name, "provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    nbf.validate(notebook)
    atomic_write_text(NOTEBOOK_PATH, nbf.writes(notebook))
    print(f"Wrote {NOTEBOOK_PATH} ({len(cells)} cells, {len(rows)} experiments)")


if __name__ == "__main__":
    main()
