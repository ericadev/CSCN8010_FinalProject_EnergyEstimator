# CSCN8010 Final Project - Energy Estimator

## Group #8
* Eris Leksi
* Erica Holden
* Reham Abuarquob


## 📌 Overview
This project focuses on **optimizing AI prompts** to reduce **energy consumption** while maintaining semantic accuracy and output quality.  
It leverages **Natural Language Processing (NLP)** models like **T5** and **Sentence-BERT (SBERT)** to:
- Shorten and improve prompts.
- Predict energy usage differences.
- Estimate potential **cost savings**.
- Assess semantic similarity between original and optimized prompts.

The system is implemented as a **Streamlit web application** following the **MVC architecture**, with backend prediction, API integration, and database support.



## 🚀 Features
- **Prompt Optimization**: Generates a shorter yet semantically equivalent version of the input.
- **Energy Prediction**: Estimates energy consumption before and after optimization.
- **Cost Savings Calculation**: Converts energy savings into $ savings based on model inference cost rates.
- **Semantic Similarity**: Measures closeness between original and optimized prompts using SBERT cosine similarity.
- **Transparency Metrics**: Displays shortening coefficient, output confidence, and energy savings %.
- **Interactive Web App**: Built with **Streamlit** for quick deployment and user interaction.



## 📊 Dataset
We use the [LLM Inference Energy Consumption Dataset](https://huggingface.co/datasets/ejhusom/llm-inference-energy-consumption) containing:
- Prompt text & characteristics.
- Token counts.
- Model parameters & inference settings.
- Measured energy usage.



## 🏗 Architecture
**Workflow**:
1. **Preprocessing** – Tokenization, cleaning, and feature extraction from prompts.
2. **Model Training**:
   - **BERT/SBERT** for semantic similarity.
   - **T5** for prompt rewriting.
   - Regression models (Linear, Polynomial, Tree-based) for energy & cost prediction.
3. **Prediction Pipeline** – Given a new prompt, output:
   - Optimized prompt
   - Energy usage (old vs. new)
   - $ savings
   - Similarity score
4. **Frontend** – Streamlit app for visualization & interaction.
5. **Persistence** – Models saved using `joblib` in the `/models` directory.



## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/your-username/sustainable-ai.git
cd sustainable-ai

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
