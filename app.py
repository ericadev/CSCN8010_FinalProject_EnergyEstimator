import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64

# --- Page Config ---
st.set_page_config(page_title="Sustainable AI - Prompt Optimizer", layout="centered")

# --- Background Image ---
def set_bg(image_path):
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

set_bg("Documents/GDC.png")

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
                response = requests.post("http://127.0.0.1:8000/api/predictor/predict", json={"prompt": prompt_input})

                if response.status_code == 200:
                    data = response.json()

                    # Generate prompt ID
                    data["prompt_id"] = len(st.session_state.history) + 1

                    # Calculate percentages
                    original_energy = float(data.get("original_energy", 1))
                    optimized_energy = float(data.get("optimized_energy", 1))
                    similarity_score = float(data.get("similarity_score", 0.0))

                    energy_pct = round(((original_energy - optimized_energy) / original_energy) * 100, 2)
                    similarity_pct = round(similarity_score * 100, 2)

                    data["energy_saved_percent"] = energy_pct
                    data["similarity_percent"] = similarity_pct

                    st.session_state.history.append(data)

                    # --- Show results ---
                    st.subheader("✅ Optimization Results")

                    # Original Prompt in Orange Box
                    st.markdown("**Original Prompt:**")
                    st.markdown(f"<div class='orange-box'>{data['original_prompt']}</div>", unsafe_allow_html=True)

                    # Optimized Prompt in Green Box
                    st.markdown("**Optimized Prompt:**")
                    st.markdown(f"<div class='green-box'>{data['optimized_prompt']}</div>", unsafe_allow_html=True)

                    # Metrics
                    col1, col2 = st.columns(2)
                    col1.metric("💡 Energy Saved", f"{energy_pct}%")
                    col2.metric("🔁 Similarity", f"{similarity_pct}%")

                    # Pie charts
                    col1, col2 = st.columns(2)

                    # Pie: Energy
                    fig_energy, ax_energy = plt.subplots(figsize=(3, 3))
                    ax_energy.pie(
                        [energy_pct, 100 - energy_pct],
                        labels=["Saved", "Used"],
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=["#67EEAF", "#F3C58A"]
                    )
                    col1.pyplot(fig_energy)

                    # Pie: Similarity
                    fig_sim, ax_sim = plt.subplots(figsize=(3, 3))
                    ax_sim.pie(
                        [similarity_pct, 100 - similarity_pct],
                        labels=["Match", "Gap"],
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=["#67EEAF", "#F3C58A"]
                    )
                    col2.pyplot(fig_sim)

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
        st.dataframe(df[["prompt_id", "original_prompt", "optimized_prompt", "energy_saved_percent", "similarity_percent"]])

    if {"original_energy", "optimized_energy", "prompt_id"}.issubset(df.columns):
        st.markdown("### 🔋 Energy Comparison")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["prompt_id"], df["original_energy"], label="Original", marker='o')
        ax1.plot(df["prompt_id"], df["optimized_energy"], label="Optimized", marker='x')
        ax1.set_xlabel("Prompt ID")
        ax1.set_ylabel("Energy (kWh)")
        ax1.legend()
        st.pyplot(fig1)
