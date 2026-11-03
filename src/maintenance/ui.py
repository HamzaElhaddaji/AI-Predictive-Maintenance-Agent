"""Run with: streamlit run src/maintenance/ui.py"""

import json
import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="FleetWatch · Maintenance", page_icon="⚙", layout="wide")
st.title("FleetWatch")
st.caption("Predictive maintenance · Engine risk, anomalies, and supporting evidence")
with st.sidebar:
    st.header("Analyze observations")
    file = st.file_uploader("Sensor data", type=["csv", "txt"])
    st.caption("26-column PHM08 text or named CSV. Leave empty to use the bundled fleet.")
    question = st.text_area("Question", "Full maintenance report", max_chars=4000)
    focus = st.number_input("Machine ID (0 = fleet)", min_value=0, step=1)
    run = st.button("Analyze fleet", type="primary", width="stretch")
    st.caption(
        "Failure estimates use a 30-cycle horizon by default. Anomaly detection is independent."
    )
if run:
    with st.spinner("Analyzing sensor history and checking evidence…"):
        try:
            data = {"question": question}
            if focus:
                data["unit_id"] = str(focus)
            files = (
                {"file": (file.name, file.getvalue(), "application/octet-stream")} if file else None
            )
            response = httpx.post(
                os.environ.get("API_URL", "http://localhost:8000") + "/report",
                data=data,
                files=files,
                timeout=180,
            )
            response.raise_for_status()
            st.session_state["report"] = response.json()
        except httpx.HTTPStatusError as exc:
            st.error(f"Analysis failed: {exc.response.json().get('detail', 'Request failed')}")
        except httpx.HTTPError:
            st.error("The maintenance API is unavailable. Start the API and try again.")
report = st.session_state.get("report")
if report:
    rows = pd.DataFrame(report["machines"])
    a, b, c = st.columns(3)
    a.metric("Machines", len(rows))
    b.metric("Failure alerts", int(rows.alert.sum()))
    c.metric("Anomalies", int(rows.anomaly.sum()))
    st.write(report["summary"])
    for warning in report["warnings"]:
        st.warning(warning)
    st.plotly_chart(
        px.bar(
            rows.sort_values("failure_probability", ascending=False).head(30),
            x=rows.sort_values("failure_probability", ascending=False).head(30).unit_id.astype(str),
            y="failure_probability",
            color="alert",
            labels={"x": "Machine", "failure_probability": "Estimated failure probability"},
            range_y=[0, 1],
        ),
        width="stretch",
    )
    st.dataframe(rows, hide_index=True, width="stretch")
    risk, evidence, quality = st.tabs(["Review actions", "Evidence", "Data quality & trace"])
    with risk:
        for action in report["actions"]:
            st.write("• " + action)
        st.caption(
            f"Alert threshold: {report['threshold']:.3f}; horizon: {report['horizon_cycles']} cycles."
        )
    with evidence:
        for uid, info in report["evidence"].items():
            with st.expander(f"Machine {uid}"):
                if "analysis" in info:
                    st.dataframe(pd.DataFrame(info["analysis"]["trends"]), hide_index=True)
                    st.caption(info["analysis"]["interpretation"])
                    st.json(info["analysis"]["correlations"])
                if "similar_machines" in info:
                    st.write("Closest training-engine snapshots (distance in scaled feature space)")
                    st.dataframe(pd.DataFrame(info["similar_machines"]), hide_index=True)
        st.subheader("Retrieved documentation")
        for citation in report["citations"]:
            st.markdown(f"**{citation['source']} · page {citation['page']}**")
            st.write(citation["text"])
        if not report["citations"]:
            st.info("No supporting documents retrieved.")
    with quality:
        st.json(report["quality"])
        st.json(report["verification"])
        st.dataframe(pd.DataFrame(report["trace"]), hide_index=True)
    st.download_button(
        "Download report",
        json.dumps(report, indent=2),
        file_name="maintenance-report.json",
        mime="application/json",
    )
else:
    st.info("Upload observations or analyze the bundled fleet to begin.")
