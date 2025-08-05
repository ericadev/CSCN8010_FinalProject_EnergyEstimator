# app.py

import streamlit as st
import requests

# Page config
st.set_page_config(page_title="Sustainable AI - Prompt Optimizer", layout="centered")
st.title("🔋 Sustainable AI – Prompt Optimizer")
st.markdown("Optimize your prompts for energy efficiency while maintaining meaning.")

# Input field
prompt_input = st.text_area("✍️ Enter your original prompt:", height=150)

# Button to trigger API call
if st.button("Optimize Prompt"):
    if not prompt_input.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Sending to the model..."):
            # Prepare the payload
            payload = {"prompt": prompt_input}

            try:
                # Send to your FastAPI backend
                response = requests.post("http://localhost:8000/predict", json=payload)

                if response.status_code == 200:
                    data = response.json()

                    # Display results
                    st.subheader("✅ Optimization Results")
                    st.markdown(f"**Original Prompt:**\n{data['original_prompt']}")
                    st.markdown(f"**Optimized Prompt:**\n{data['optimized_prompt']}")
                    st.markdown(f"**Token Count:** `{data['token_count']}`")
                    st.markdown(f"**Energy Saved:** `{data['energy_saved']} kWh`")
                    st.markdown(f"**Semantic Similarity:** `{round(data['similarity_score'], 3)}`")

                else:
                    st.error(f"API Error: {response.status_code} – {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
