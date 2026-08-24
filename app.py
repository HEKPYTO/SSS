"""app.py — Streamlit demo: upload ARFF -> frontier + selected features"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st

from src.sss.arff import load_arff
from src.sss.sss import SFFS

st.title("SSS sSFFS — 10k-d budgeted feature selection")
uploaded = st.file_uploader("ARFF (dense/sparse, numeric + nominal class)", type=["arff"])
d_val = st.slider("target d", 1, 100, 10)
y_val = st.slider("y (forward budget)", 5, 200, 25)
if uploaded:
    path = f"/tmp/{uploaded.name}"
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    data = load_arff(path)
    st.write(
        f"Loaded {data.n_features} features, {data.n_classes} classes, {sum(data.class_size)} samples"
    )
    if st.button("Run sSFFS"):
        with st.spinner("Searching..."):
            sffs = SFFS(
                y=y_val,
                y_back=max(5, y_val // 2),
                rho_u=0.2,
                tau=0.0,
                warmup_probes=100,
                warmup_card=10,
                delta=5,
                seed=1,
                scaler="void",
            )
            sel = sffs.fit_filter(data, target_d=d_val, train_pct=50, test_pct=40)
            st.success(
                f"Selected {len(sel)} features, value {sffs.value_:.4f}, evals {sffs.evaluations_}"
            )
            st.write(sel)
            # crude frontier: run for d=1..target and plot value (reuse evaluations would be heavy, so single run)
            st.line_chart({"selected": sel})
else:
    st.info("Upload anc/data/reuters_apte.arff to try")
