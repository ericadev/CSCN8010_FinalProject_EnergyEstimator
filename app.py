import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64
from typing import Any, Dict, Iterable, Optional

# =========================
# Config
# =========================
st.set_page_config(page_title="Sustainable AI - Prompt Optimizer", layout="centered")

COST_PER_KWH = 0.284  # 28.4¢ per kWh (stored as dollars)
CURRENCY = "$"        # totals use dollars
DEBUG = False         # True -> show raw API JSON

# =========================
# Background + Base Styles
# =========================
def set_bg(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
            }}
            .green-box {{
                background-color: #d9fdd3;
                border: 1px solid #a6d8a8;
                padding: 10px;
                border-radius: 8px;
                font-size: 16px;
            }}
            .orange-box {{
                background-color: #fff3cd;
                border: 1px solid #ffeeba;
                padding: 10px;
                border-radius: 8px;
                font-size: 16px;
            }}

            /* ===== KPI cards (uniform) ===== */
            .kpi-card {{
              display:flex; flex-direction:column; align-items:center; justify-content:center;
              gap:.35rem; padding:.8rem 1rem; border-radius:14px;
              background:rgba(255,255,255,.75); border:1px solid rgba(0,0,0,.06);
              min-height:96px; text-align:center;
            }}
            .kpi-label {{ font-size:.90rem; color:#4b5563; font-weight:600; }}
            .kpi-value {{ font-size:1.6rem; font-weight:700; line-height:1; }}

            /* Confidence badge inside KPI */
            .conf-badge {{
              display:inline-flex; align-items:center; gap:.45rem;
              padding:.2rem .55rem; border-radius:999px; font-weight:600;
              border:1px solid rgba(0,0,0,.08); white-space:nowrap;
            }}
            .conf-high   {{ background:#EAF7EF; color:#166534; }}
            .conf-medium {{ background:#FFF4E5; color:#92400E; }}
            .conf-low    {{ background:#FEE2E2; color:#7F1D1D; }}
            .conf-dot {{ width:.55rem; height:.55rem; border-radius:50%; display:inline-block; }}

            .section-title {{ text-align:center; font-weight:800; margin:.2rem 0 .5rem 0; }}
        </style>
        """, unsafe_allow_html=True)
    except Exception:
        pass

set_bg("Documents/GDC.png")

# =========================
# Helpers
# =========================
def render_energy_bar(label: str, value: float, max_value: float, color_hex: str):
    pct = 0 if max_value == 0 else min(100, (value / max_value) * 100)
    st.markdown(f"""
    <div style="margin:6px 0 16px 0;">
        <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:4px;">
            <span><b>{label} Energy</b></span>
            <span>{value:.4f} kWh</span>
        </div>
        <div style="height:12px; width:100%; background:#eee; border-radius:6px; overflow:hidden;">
            <div style="height:12px; width:{pct:.1f}%; background:{color_hex};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def energy_band(e: float) -> str:
    return "High" if e > 0.07 else ("Medium" if e > 0.03 else "Low")

def format_currency(amount: float) -> str:
    return f"{CURRENCY}{amount:,.2f}"  # dollars for totals

def _find_key_case_insensitive(payload: Any, keys: Iterable[str]) -> Optional[Any]:
    if isinstance(payload, dict):
        lower_map = {k.lower(): k for k in payload.keys()}
        for target in keys:
            if target.lower() in lower_map:
                return payload[lower_map[target.lower()]]
        for v in payload.values():
            found = _find_key_case_insensitive(v, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_key_case_insensitive(item, keys)
            if found is not None:
                return found
    return None

def _normalize_confidence_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        score = float(value)
        if score > 1.0:
            score = score / 100.0
        if score >= 0.85: return "High"
        if score >= 0.75: return "Medium"
        return "Low"
    s = str(value).strip().lower()
    if s in {"high", "h", "strong"}: return "High"
    if s in {"medium", "med", "moderate", "mid"}: return "Medium"
    if s in {"low", "l", "weak"}: return "Low"
    return None

def fallback_confidence_from_metrics(energy_saved_pct: float, similarity_score: float) -> str:
    if energy_saved_pct >= 15 and similarity_score >= 0.85: return "High"
    if energy_saved_pct >= 5 and similarity_score >= 0.75: return "Medium"
    return "Low"

def get_confidence_label(api_payload: Dict[str, Any], energy_saved_pct: float, similarity_score: float) -> str:
    keys = ("status", "confidence", "confidence_status", "model_status", "confidence_label")
    raw = _find_key_case_insensitive(api_payload, keys)
    label = _normalize_confidence_value(raw)
    if label is None:
        for k in ("result", "meta", "details", "prediction"):
            sub = api_payload.get(k)
            if sub is not None:
                raw2 = _find_key_case_insensitive(sub, keys)
                label = _normalize_confidence_value(raw2)
                if label is not None:
                    break
    if label is None:
        label = fallback_confidence_from_metrics(energy_saved_pct, similarity_score)
    return label

# =========================
# UI
# =========================
st.markdown("<h1 style='text-align: center;'>🔋 Sustainable AI – Prompt Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Optimize your prompts for energy efficiency while maintaining meaning.</p>", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

prompt_input = st.text_area("✍️ Enter your original prompt:", height=150)

# =========================
# Action
# =========================
if st.button("Optimize Prompt"):
    if not prompt_input.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Sending to the model..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/api/predictor/predict",
                    json={"prompt": prompt_input},
                    timeout=60
                )
                response_by_verb_count = requests.post(
                    "http://127.0.0.1:8000/api/predictor/predict_by_verbs",
                    json={"prompt": prompt_input},
                    timeout=60
                )
                if response.status_code == 200 and response_by_verb_count.status_code == 200:
                    data = response.json()
                    
                    if DEBUG:
                        st.expander("🔍 Raw API response (debug)").json(data)

                    data["prompt_id"] = len(st.session_state.history) + 1

                    # Extract
                    original_energy  = float(data.get("original_energy", 0.0))
                    optimized_energy = float(data.get("optimized_energy", 0.0))
                    similarity_score = float(data.get("similarity_score", 0.0))

                    # Derived
                    energy_saved_kwh = max(0.0, response_by_verb_count.json())
                    energy_pct       = 0.0 if original_energy <= 0 else round(((original_energy - optimized_energy)/original_energy)*100, 2)
                    similarity_pct   = round(similarity_score * 100, 2)
                    money_saved_usd  = energy_saved_kwh * COST_PER_KWH

                    # Confidence
                    confidence_label = get_confidence_label(data, energy_pct, similarity_score)
                    conf_class = {"High":"conf-high", "Medium":"conf-medium", "Low":"conf-low"}.get(confidence_label, "conf-medium")
                    conf_dot_color = {"High":"#16a34a", "Medium":"#f59e0b", "Low":"#ef4444"}.get(confidence_label, "#f59e0b")

                    # Persist
                    data.update({
                        "energy_saved_kwh":    energy_saved_kwh,
                        "energy_saved_percent":energy_pct,
                        "similarity_percent":  similarity_pct,
                        "saved_money":         money_saved_usd,  # store dollars
                        "confidence":          confidence_label
                    })
                    st.session_state.history.append(data)

                    # =========================
                    # Results
                    # =========================
                    st.subheader("✅ Optimization Results")

                    st.markdown("**Original Prompt:**")
                    st.markdown(f"<div class='orange-box'>{data.get('original_prompt','')}</div>", unsafe_allow_html=True)
                    max_energy = max(original_energy, optimized_energy, 1e-9)
                    render_energy_bar("Original", original_energy, max_energy, "#F3C58A")

                    st.markdown("**Optimized Prompt:**")
                    st.markdown(f"<div class='green-box'>{data.get('optimized_prompt','')}</div>", unsafe_allow_html=True)
                    render_energy_bar("Optimized", optimized_energy, max_energy, "#67EEAF")

                    st.caption(f"⚡ Original: **{energy_band(original_energy)}**  |  ✅ Optimized: **{energy_band(optimized_energy)}**")

                    # ===== KPI ROW (uniform cards) =====
                    colA, colB, colC, colD = st.columns(4)

                    with colA:
                        st.markdown(
                            '<div class="kpi-card">'
                            '<div class="kpi-label">💡 Energy Saved</div>'
                            f'<div class="kpi-value">{energy_pct}%</div>'
                            '</div>', unsafe_allow_html=True)

                    with colB:
                        st.markdown(
                            '<div class="kpi-card">'
                            '<div class="kpi-label">🔁 Similarity</div>'
                            f'<div class="kpi-value">{similarity_pct}%</div>'
                            '</div>', unsafe_allow_html=True)

                    with colC:
                        cents = money_saved_usd * 100.0
                        st.markdown(
                            '<div class="kpi-card">'
                            '<div class="kpi-label">💰 Money Saved</div>'
                            f'<div class="kpi-value">{cents:.2f}¢</div>'
                            '</div>', unsafe_allow_html=True)

                    with colD:
                        st.markdown(
                            '<div class="kpi-card">'
                            '<div class="kpi-label">🔒 Confidence</div>'
                            f'<div class="kpi-value"><span class="conf-badge {conf_class}">'
                            f'<span class="conf-dot" style="background:{conf_dot_color};"></span>'
                            f'{confidence_label}'
                            '</span></div>'
                            '</div>', unsafe_allow_html=True)

                    # ===== Charts (consistent size) =====
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 💡 Energy Saved", unsafe_allow_html=True)
                        fig_energy, ax_energy = plt.subplots(figsize=(4, 4))
                        ax_energy.pie(
                            [max(0, energy_pct), max(0, 100 - energy_pct)],
                            labels=["Saved", "Used"],
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=["#67EEAF", "#F3C58A"],
                            textprops={'fontsize': 12}
                        )
                        ax_energy.axis('equal')
                        st.pyplot(fig_energy, use_container_width=False)

                    with col2:
                        st.markdown("### 🔁 Similarity", unsafe_allow_html=True)
                        fig_sim, ax_sim = plt.subplots(figsize=(4, 4))
                        ax_sim.pie(
                            [similarity_pct, max(0, 100 - similarity_pct)],
                            labels=["Match", "Gap"],
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=["#67EEAF", "#F3C58A"],
                            textprops={'fontsize': 12}
                        )
                        ax_sim.axis('equal')
                        st.pyplot(fig_sim, use_container_width=False)

                else:
                    st.error(f"API Error: {response.status_code} – {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# =========================
# History / Analytics
# =========================
if st.session_state.history:
    st.markdown("---")
    st.subheader("📈 Optimization History")

    df = pd.DataFrame(st.session_state.history)

    with st.expander("🧾 View Prompt History Table"):
        cols_to_show = [
            "prompt_id", "original_prompt", "optimized_prompt",
            "original_energy", "optimized_energy",
            "energy_saved_kwh", "energy_saved_percent",
            "similarity_percent", "saved_money", "confidence"
        ]
        present = [c for c in cols_to_show if c in df.columns]
        df_display = df.copy()
        if "saved_money" in df_display.columns:
            df_display["saved_money_cents"] = (df_display["saved_money"] * 100).round(2)
        table_cols = [c for c in present if c != "saved_money"]
        if "saved_money_cents" in df_display.columns:
            table_cols.append("saved_money_cents")
        st.dataframe(df_display[table_cols])

    # Energy over time
    if {"original_energy", "optimized_energy", "prompt_id"}.issubset(df.columns):
        st.markdown("### 🔋 Energy Comparison Over Time")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["prompt_id"], df["original_energy"], label="Original", marker='o')
        ax1.plot(df["prompt_id"], df["optimized_energy"], label="Optimized", marker='x')
        ax1.set_xlabel("Prompt ID")
        ax1.set_ylabel("Energy (kWh)")
        ax1.legend()
        st.pyplot(fig1)

    # Money saved per prompt (cents)
    if {"saved_money", "prompt_id"}.issubset(df.columns):
        st.markdown("### 💰 Money Saved per Prompt (¢)")
        fig2, ax2 = plt.subplots()
        ax2.bar(df["prompt_id"], df["saved_money"] * 100.0)
        ax2.set_xlabel("Prompt ID")
        ax2.set_ylabel("Money Saved (¢)")
        st.pyplot(fig2)

        # Cumulative savings (dollars)
        st.markdown("### 📉 Cumulative Money Saved (Dollars)")
        df_sorted = df.sort_values("prompt_id")
        df_sorted["cumulative_money_saved"] = df_sorted["saved_money"].cumsum()
        fig3, ax3 = plt.subplots()
        ax3.plot(df_sorted["prompt_id"], df_sorted["cumulative_money_saved"], marker='o')
        ax3.set_xlabel("Prompt ID")
        ax3.set_ylabel("Cumulative Savings ($)")
        st.pyplot(fig3)

        total_saved = float(df["saved_money"].sum())
        st.success(f"Total money saved so far: **{format_currency(total_saved)}**")
