# 🌱 EcoPrompt AI – Sustainable Prompt Engineering for Energy Efficiency

## Group #8
* Eris Leksi (9067882)
* Erica Holden (5490685)
* Reham Abuarquob (9062922)
 

## Everything from the presentation is explained down here:

## 📌 Overview
This project focuses on **optimizing AI prompts** to reduce **energy consumption** while maintaining semantic accuracy and output quality.  
It leverages **Natural Language Processing (NLP)** models like **T5** and **Sentence-BERT (SBERT)** to:
- Shorten and improve prompts.
- Predict energy usage differences.
- Estimate potential **cost savings**.
- Assess semantic similarity between original and optimized prompts.

The system is implemented as a **Streamlit web application** following the **MVC architecture**, with backend prediction, API integration, and database support.

## 🛑 Our Problem
AI models are becoming increasingly energy-intensive, creating both **environmental challenges** and **regulatory pressures**.  
Our project addresses the urgent need to **measure, report, and optimize** AI energy usage.

**Key Challenges:**
- ⚠ **1,240 AI data centers** projected to be built in 2025.
- 💧 **Severe water usage**, especially in dry regions.
- 💡 **EU regulation (2026):** Mandatory AI energy reporting.
- 💰 **$9.2B** in health-related costs from AI emissions.

**Our Goal:**  
Enable **transparent, efficient AI usage** with smart prompt/context engineering and real-time energy insights.


## 💼 Business Implementation

This project is designed to integrate seamlessly into real-world AI workflows, enabling organizations to **optimize prompts**, **reduce energy usage**, and **maintain high-quality outputs**.

**Workflow:**
1. **User Interaction** – A user submits a request through an application or interface.
2. **Prompt Generation** – Initial prompts are created for AI processing.
3. **Prompt Optimization** – Our system refines prompts using smart engineering to reduce token count while preserving meaning.
4. **Energy & Cost Tracking** – Energy usage is monitored and estimated in real-time.
5. **AI Execution** – The optimized prompt is processed through large models such as **ChatGPT**, **Gemini**, or **Copilot**.
6. **Insights & Reporting** – Energy savings, semantic similarity, and performance metrics are shared with stakeholders for transparency.
7. **Business Integration** – Results are used to improve efficiency in operations, marketing, customer service, and more.


## 🌍 Relevance

### **Social & Economic**
- Make chatbots and virtual assistants **more affordable and sustainable**.
- Reduce the **carbon footprint** of messaging and search.
- Improve **accessibility** of NLPs in regions with limited computing resources.
- Promote **sustainable computing** in alignment with global climate goals.

### **Industrial**
- Reduce **compute time** on platforms like AWS, Azure, GCP, lowering operational costs.
- Prepare industries for **upcoming environmental regulations** on AI energy usage and carbon tracking.
- Enable **more queries per second** on existing infrastructure without extra hardware.
- Lower server workload, reducing cooling demands and **extending hardware lifespan**.


## 🚀 Features
- **Prompt Optimization**: Generates a longer yet more energy efficient prompt than the input one.
- **Energy Prediction**: Estimates energy consumption before and after optimization.
- **Cost Savings Calculation**: Converts energy savings into $ savings based on Province of Ontario average electricity rate.
- **Semantic Similarity**: Measures closeness between original and optimized prompts using SBERT cosine similarity.
- **Transparency Metrics**: Displays shortening coefficient, output confidence, energy savings %, $ savings.
- **Interactive Web App**: Built with **Streamlit** for quick deployment and user interaction.


## 💡 Why This Project Matters

AI models consume significant energy, leading to high costs and environmental impact.  
This project reduces that footprint by **optimizing prompts** to lower computation while preserving meaning, tracking energy use, and promoting **sustainable, cost-efficient AI**.


## 📊 Dataset
We use the [LLM Inference Energy Consumption Dataset](https://huggingface.co/datasets/ejhusom/llm-inference-energy-consumption) containing:
- Prompt text & characteristics.
- Token counts.
- Model parameters & inference settings.
- Measured energy usage.
- etc


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


## 🏗 Architecture Diagram

The system is designed with a modular architecture to allow **scalable AI energy optimization**:

1. **User Interaction** – Users submit prompts via the Streamlit web application.
2. **API Server** –  
   - Endpoints:  
     - `/predict` – Main prediction route.  
     - `/predict_by_verbs` – Prediction based on instruction complexity.  
   - Controller – Routes requests to the correct processing service.  
   - Predictor Service – Handles preprocessing, optimization, and model calls.
3. **ML Models** –  
   - **BART Class** – Performs prompt rephrasing for optimization.  
   - **SBERT Class** – Computes semantic similarity between original and optimized prompts.  
   - **Exponential Offset Model** – Predicts energy savings based on prompt features.


## 🔄 NLP Pipeline

Our system follows a structured **NLP pipelining process** to transform raw text into optimized prompts with measurable energy savings:

1. **Text Acquisition** – Ingest raw text from sources and capture metadata.  
2. **Text Preprocessing / Cleaning** – Lowercasing, trimming whitespace, and Unicode normalization.  
3. **Tokenization & Linguistic Analysis** – Segment and annotate text.  
4. **Vectorization** – Generate contextual embeddings for meaning-aware models.  
5. **Feature Engineering** – Compute readability and sentiment scores.  
6. **Modeling** – Apply **BART** for rephrasing, **Random Forest** for energy prediction, and **SBERT** for semantic similarity.  
7. **Evaluation** – Measure energy saved (%), cost saved, similarity, and confidence.  
8. **Deployment** – FastAPI backend with a Streamlit frontend.


## 📂 Project Structure
CSCN8010_FinalProject_EnergyEstimator
|  Documents/
│ └── GDC.png # Project diagram or related image
│
├── api/ # API-related code
│ ├── controllers/ # Controllers handle incoming API requests
│ │ └── predict_controller.py # Logic for prediction API endpoints
│ ├── models/ # Data models for API requests/responses
│ │ └── predict_request.py # Request model for prediction inputs
│ └── services/ # Backend service logic
│ ├── init.py
│ ├── main.py # Entry point for service execution
│ └── requirements.txt # Dependencies for the service module
│
├── data/ # Dataset files
│ ├── alpaca_llama3_70b_server.csv # Model output dataset
│ └── improved_prompts.csv # Optimized prompts dataset
│
├── models/ # Saved ML models
│ ├── sbert_model/ # SBERT model directory
│ │ ├── 1_Pooling/ # Pooling configuration
│ │ │ ├── config.json
│ │ │ └── README.md
│ │ ├── config.json
│ │ ├── config_sentence_transformers.json
│ │ ├── model.safetensors
│ │ ├── modules.json
│ │ ├── sentence_bert_config.json
│ │ ├── special_tokens_map.json
│ │ ├── tokenizer.json
│ │ ├── tokenizer_config.json
│ │ ├── vocab.txt
│ │ ├── init.py
│ │ ├── bart_model.pkl # Trained BART model
│ │ ├── energy_model_rf.pkl # Random Forest model for energy prediction
│ │ ├── exp_offset_model.pkl # Exponential offset model
│ │ └── sbert_model.pkl # Trained SBERT model
│
├── notebooks/ # Jupyter Notebooks for experiments
│ ├── NLP_script.ipynb # NLP processing and training
│ └── prediction_script.ipynb # Prediction testing and evaluation
│
├── scripts/ # Utility or helper scripts
│
├── .gitattributes
├── .gitignore
├── README.md # Project documentation
├── app.py # Main application script
├── exp_offset_model.py # Exponential offset model implementation
├── integration.py # Integration logic for components
└── requirements.txt # Project dependencies


## 📂 Project Structure – Explanation

- **Documents/** – Contains supporting documentation or visuals for the project (e.g., diagrams, images).
  - `GDC.png` – Graphic or diagram used in reports/presentations.

- **api/** – API layer for serving predictions and handling requests.
  - **controllers/** – Functions that process incoming HTTP requests and call the appropriate services.
    - `predict_controller.py` – Handles prediction-related API endpoints.
  - **models/** – Defines data structures for requests/responses.
    - `predict_request.py` – Schema for input data to the prediction API.
  - **services/** – Core business logic for processing predictions.
    - `main.py` – API entry point.
    - `__init__.py` – Marks the folder as a Python package.
    - `requirements.txt` – Python dependencies specific to the API service.

- **data/** – Raw and processed datasets used for training/testing.
  - `alpaca_llama3_70b_server.csv` – Dataset with baseline prompts.
  - `improved_prompts.csv` – Dataset with optimized prompts.

- **models/** – Stored machine learning models and configurations.
  - **sbert_model/** – Saved Sentence-BERT model and configuration files.
    - **1_Pooling/** – Pooling layer configuration.
    - Various `.json` and `.txt` – Model settings, tokenizer, and vocab.
    - `.pkl` files – Serialized trained models:
      - `bart_model.pkl` – BART model for prompt rephrasing.
      - `energy_model_rf.pkl` – Random Forest model for energy prediction.
      - `exp_offset_model.pkl` – Exponential offset model for energy estimation.
      - `sbert_model.pkl` – SBERT model for semantic similarity.
  
- **notebooks/** – Jupyter notebooks for experiments and analysis.
  - `NLP_script.ipynb` – NLP preprocessing and training pipeline.
  - `prediction_script.ipynb` – Testing predictions and model performance.

- **scripts/** – Additional helper or automation scripts.

- **app.py** – Main application script, possibly running the Streamlit frontend.
- **exp_offset_model.py** – Implementation of the exponential offset energy model.
- **integration.py** – Code integrating multiple components (models, API, frontend).
- **requirements.txt** – Master list of Python dependencies.
- **.gitattributes / .gitignore** – Git configuration files.
- **README.md** – Project documentation.

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
