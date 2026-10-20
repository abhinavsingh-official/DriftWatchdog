"""DriftWatchdog - watches a deployed classifier for data drift and
tracks its real accuracy as synthetic drift increases."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from drift_injector import DRIFT_KINDS, inject_drift
from model_shadow import load_model, load_reference_and_holdout, score_model
from drift_watchdog import FeatureDriftWatchdog

st.set_page_config(page_title="DriftWatchdog", layout="wide")

PAGE_STYLE = """
<style>
.block-container { padding-top: 2rem; max-width: 1150px; }
h1 { font-weight: 700; letter-spacing: -0.02em; color: #b91c1c; }
.dw-panel {
    background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.dw-metric { font-size: 2rem; font-weight: 700; }
.dw-metric-label { color: #4b5563; font-size: 0.9rem; }
.dw-ok { color: #15803d; }
.dw-alert { color: #b91c1c; }
</style>
"""


@st.cache_resource
def _load():
    model = load_model()
    (X_ref, y_ref), (X_hold, y_hold), feature_cols = load_reference_and_holdout()
    watchdog = FeatureDriftWatchdog(feature_names=feature_cols).fit(X_ref)
    baseline_acc = score_model(model, X_hold, y_hold, feature_cols)
    return model, X_hold, y_hold, feature_cols, watchdog, baseline_acc


def intro_tab():
    st.markdown(
        '<div class="dw-panel" style="padding:2rem 2.2rem;">'
        "<h2 style='margin-top:0;'>A watchdog for a model that's already shipped</h2>"
        "<p style='color:#374151; font-size:1.05rem; max-width:780px; line-height:1.6;'>"
        "Most portfolio ML projects stop at training a model. This one assumes the model "
        "is already deployed and asks the next question: how would you actually know if "
        "the data it's seeing in production has quietly drifted away from what it was "
        "trained on - before accuracy visibly tanks?"
        "</p>"
        "<p style='color:#374151; font-size:1.05rem; max-width:780px; line-height:1.6;'>"
        "It wraps the real MLP emotion classifier from "
        "<a href='https://github.com/abhinavsingh-official/EmotionDetect' target='_blank'>EmotionDetect</a> "
        "and its real 180-feature training distribution, injects synthetic production drift "
        "(a sensor change, added noise, a shifting population, a broken upstream feature), "
        "detects it statistically with a per-feature Kolmogorov-Smirnov test "
        "(<a href='https://github.com/ml-cube/ml3-drift' target='_blank'>ml3-drift</a>, "
        "Bonferroni-corrected across 180 features), and shows the model's real accuracy "
        "degrading on the same drifted batch - not a faked number."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#4b5563;">Head to <b>Watchdog</b> for a single drifted batch, or '
        "<b>Timeline</b> to watch drift and accuracy move together as it ramps up over "
        "simulated production batches.</p>",
        unsafe_allow_html=True,
    )


def watchdog_tab():
    model, X_hold, y_hold, feature_cols, watchdog, baseline_acc = _load()

    col1, col2 = st.columns(2)
    with col1:
        kind = st.selectbox("Drift kind", DRIFT_KINDS, index=1)
    with col2:
        intensity = st.slider("Drift intensity", 0.0, 1.0, 0.0, 0.05)

    X_batch = inject_drift(X_hold, intensity=intensity, kind=kind)
    report = watchdog.detect(X_batch)
    live_acc = score_model(model, X_batch, y_hold, feature_cols)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="dw-panel"><div class="dw-metric">{report.n_drifted}/{report.n_features}</div>'
            '<div class="dw-metric-label">Features flagged as drifted</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        delta = live_acc - baseline_acc
        acc_class = "dw-ok" if delta > -0.02 else "dw-alert"
        st.markdown(
            f'<div class="dw-panel"><div class="dw-metric {acc_class}">{live_acc:.0%}</div>'
            f'<div class="dw-metric-label">Live accuracy (baseline {baseline_acc:.0%})</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="dw-panel"><div class="dw-metric">{report.corrected_p_value:.1e}</div>'
            '<div class="dw-metric-label">Per-feature p-value threshold (Bonferroni-corrected)</div></div>',
            unsafe_allow_html=True,
        )

    if report.n_drifted > 0:
        st.warning(
            f"Drift detected in {report.n_drifted} of {report.n_features} features: "
            + ", ".join(report.drifted_features[:15])
            + (" ..." if report.n_drifted > 15 else "")
        )
    else:
        st.success("No statistically significant drift detected in this batch.")


def timeline_tab():
    model, X_hold, y_hold, feature_cols, watchdog, baseline_acc = _load()

    col1, col2 = st.columns(2)
    with col1:
        kind = st.selectbox("Drift kind", DRIFT_KINDS, index=1, key="timeline_kind")
    with col2:
        n_batches = st.slider("Simulated production batches", 5, 30, 12)

    if not st.button("Run simulation"):
        return

    intensities = np.linspace(0.0, 1.0, n_batches)
    fractions, accuracies = [], []
    for i, intensity in enumerate(intensities):
        X_batch = inject_drift(X_hold, intensity=float(intensity), kind=kind, seed=i)
        report = watchdog.detect(X_batch)
        fractions.append(report.fraction_drifted)
        accuracies.append(score_model(model, X_batch, y_hold, feature_cols))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n_batches)), y=accuracies, name="Model accuracy", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=list(range(n_batches)), y=fractions, name="Fraction of features drifted",
                              line=dict(color="#b91c1c"), yaxis="y2"))
    fig.update_layout(
        xaxis_title="Simulated production batch (drift ramping up over time)",
        yaxis=dict(title="Model accuracy", range=[0, 1.05]),
        yaxis2=dict(title="Fraction of features drifted", overlaying="y", side="right", range=[0, 1.05]),
        legend=dict(orientation="h", y=1.12),
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Both lines come from the same real model and real statistical test run against each "
        "progressively more drifted batch - the watchdog's drift signal and the model's actual "
        "accuracy degrade together, not independently simulated."
    )


def main():
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    st.title("DriftWatchdog")
    st.caption("Statistical drift detection and real accuracy tracking for a deployed classifier.")
    tab0, tab1, tab2 = st.tabs(["Home", "Watchdog", "Timeline"])
    with tab0:
        intro_tab()
    with tab1:
        watchdog_tab()
    with tab2:
        timeline_tab()


if __name__ == "__main__":
    main()
