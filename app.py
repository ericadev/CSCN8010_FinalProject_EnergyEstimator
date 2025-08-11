import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64

# --- Page Config ---
st.set_page_config(page_title="Sustainable AI - Prompt Optimizer", layout="centered")

# --- Background Image ---
def set_bg(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        css = f"""
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
                border-radius: 5px;
                font-size: 16px;
            }}
            .orange-box {{
                background-color: #fff3cd;
                border: 1px solid #ffeeba;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
            }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass  # Don't block app if background is missing

set_bg("Documents/GDC.png")

# --- Energy bar helper ---
def render_energy_bar(label: str, value: float, max_value: float, color_hex: str):
    pct = 0 if max_value == 0 else min(100, (value / max_value) * 100)
    bar_html = f"""
    <div style="margin:6px 0 16px 0;">
        <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:4px;">
            <span><b>{label} Energy</b></span>
            <span>{value:.4f} kWh</span>
        </div>
        <div style="height:12px; width:100%; background:#eee; border-radius:6px; overflow:hidden;">
            <div style="height:12px; width:{pct:.1f}%; background:{color_hex};"></div>
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

def energy_band(e: float) -> str:
    return "High" if e > 0.07 else ("Medium" if e > 0.03 else "Low")

# --- Title and Intro ---
st.markdown("<h1 style='text-align: center;'>🔋 Sustainable AI – Prompt Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Optimize your prompts for energy efficiency while maintaining meaning.</p>", unsafe_allow_html=True)

# --- Initialize session state ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Prompt Input ---
prompt_input = st.text_area("✍️ Enter your original prompt:", height=150)

# --- Button and API Call ---
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

                if response.status_code == 200:
                    data = response.json()

                    # Generate prompt ID
                    data["prompt_id"] = len(st.session_state.history) + 1

                    # Extract values with safe defaults
                    original_energy = float(data.get("original_energy", 0.0))
                    optimized_energy = float(data.get("optimized_energy", 0.0))
                    similarity_score = float(data.get("similarity_score", 0.0))

                    # Derived metrics
                    energy_pct = 0.0
                    if original_energy > 0:
                        energy_pct = round(((original_energy - optimized_energy) / original_energy) * 100, 2)
                    similarity_pct = round(similarity_score * 100, 2)

                    data["energy_saved_percent"] = energy_pct
                    data["similarity_percent"] = similarity_pct

                    st.session_state.history.append(data)

                    # --- Show results ---
                    st.subheader("✅ Optimization Results")

                    # Original Prompt in Orange Box
                    st.markdown("**Original Prompt:**")
                    st.markdown(f"<div class='orange-box'>{data.get('original_prompt','')}</div>", unsafe_allow_html=True)

                    # Energy bar under Original
                    max_energy = max(original_energy, optimized_energy, 1e-9)
                    render_energy_bar("Original", original_energy, max_energy, "#F3C58A")

                    # Optimized Prompt in Green Box
                    st.markdown("**Optimized Prompt:**")
                    st.markdown(f"<div class='green-box'>{data.get('optimized_prompt','')}</div>", unsafe_allow_html=True)

                    # Energy bar under Optimized
                    render_energy_bar("Optimized", optimized_energy, max_energy, "#67EEAF")

                    # Quick labels + key metrics
                    st.caption(
                        f"⚡ Original: **{energy_band(original_energy)}**  |  ✅ Optimized: **{energy_band(optimized_energy)}**"
                    )
                    colm1, colm2 = st.columns(2)
                    colm1.metric("💡 Energy Saved", f"{energy_pct}%")
                    colm2.metric("🔁 Similarity", f"{similarity_pct}%")

                    # =========================
                    # Bigger side-by-side pies
                    # =========================
                    col1, col2 = st.columns([1.5, 1.5])  # wider columns than default

                    with col1:
                        st.markdown("### 💡 Energy Saved")
                        fig_energy, ax_energy = plt.subplots(figsize=(6, 6))
                        ax_energy.pie(
                            [max(0, energy_pct), max(0, 100 - energy_pct)],
                            labels=["Saved", "Used"],
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=["#67EEAF", "#F3C58A"],
                            textprops={'fontsize': 14}
                        )
                        ax_energy.axis('equal')
                        st.pyplot(fig_energy, use_container_width=True)

                    with col2:
                        st.markdown("### 🔁 Similarity")
                        fig_sim, ax_sim = plt.subplots(figsize=(6, 6))
                        ax_sim.pie(
                            [similarity_pct, max(0, 100 - similarity_pct)],
                            labels=["Match", "Gap"],
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=["#67EEAF", "#F3C58A"],
                            textprops={'fontsize': 14}
                        )
                        ax_sim.axis('equal')
                        st.pyplot(fig_sim, use_container_width=True)

                else:
                    st.error(f"API Error: {response.status_code} – {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# --- History Dashboard ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("📈 Optimization History")

    df = pd.DataFrame(st.session_state.history)

    with st.expander("🧾 View Prompt History Table"):
        cols_to_show = [
            "prompt_id", "original_prompt", "optimized_prompt",
            "original_energy", "optimized_energy",
            "energy_saved_percent", "similarity_percent"
        ]
        present = [c for c in cols_to_show if c in df.columns]
        st.dataframe(df[present])

    if {"original_energy", "optimized_energy", "prompt_id"}.issubset(df.columns):
        st.markdown("### 🔋 Energy Comparison Over Time")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["prompt_id"], df["original_energy"], label="Original", marker='o')
        ax1.plot(df["prompt_id"], df["optimized_energy"], label="Optimized", marker='x')
        ax1.set_xlabel("Prompt ID")
        ax1.set_ylabel("Energy (kWh)")
        ax1.legend()
        st.pyplot(fig1)
