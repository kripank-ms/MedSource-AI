# 🏥 MedSource AI

> **AI-powered medical assistance, prediction, and medical image analysis platform**

MedSource AI is a full-stack AI application built with **Python, FastAPI, Machine Learning, Deep Learning, Computer Vision, and Retrieval-Augmented Generation (RAG)**.

The platform brings multiple medical AI capabilities together in one application, including disease-risk prediction, medical image classification, and an AI-powered medical chatbot.

---

## ✨ Features

| Feature | Technology |
|---|---|
| 🩺 Disease prediction | Scikit-learn / Joblib |
| 🩻 Medical image analysis | TensorFlow / Keras |
| 🧠 Brain MRI analysis | Deep Learning |
| 🫁 Chest CT analysis | Deep Learning |
| 🫘 Kidney CT analysis | Deep Learning |
| 🩻 X-Ray classification | Deep Learning |
| 💬 Medical AI chatbot | RAG + Ollama |
| 🔎 Document retrieval | ChromaDB |
| 🧮 Text embeddings | Sentence Transformers |
| 🔐 User authentication | FastAPI |
| 🌐 Backend API | FastAPI |
| 🎨 Web interface | HTML + CSS + JavaScript |

---

## 🧠 System Architecture

```text
                         ┌─────────────────────┐
                         │       User          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    MedSource AI     │
                         │     Web Interface   │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                ┌────────────────┐    ┌────────────────┐
                │ Disease Models │    │ Image Analysis │
                │ ML / Joblib    │    │ CNN / Keras    │
                └───────┬────────┘    └───────┬────────┘
                        │                     │
                        ▼                     ▼
                ┌────────────────┐    ┌────────────────┐
                │    Prediction  │    │   Prediction   │
                └────────────────┘    └────────────────┘

                         ┌─────────────────────┐
                         │   Medical Chatbot   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Embedding + Search  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      ChromaDB        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       Reranker       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Ollama LLM      │
                         └─────────────────────┘
```

---

# 🩺 Disease Prediction

MedSource AI includes machine-learning-based prediction modules for different health conditions.

Current model files include modules such as:

- ❤️ Heart Disease
- 🫁 COPD
- 🧬 HIV Risk
- 🧠 Stroke
- 🫘 CKD / Kidney Disease
- 🦠 Tuberculosis

Typical workflow:

User Input
    │
    ▼
Input Validation
    │
    ▼
Preprocessing
    │
    ▼
Trained ML Model
    │
    ▼
Prediction
    │
    ▼
Result

The prediction models are intended for **educational and research purposes** and should not be used as standalone medical diagnosis tools.



# 🩻 Medical Image Analysis

The application contains multiple deep-learning models for medical image analysis.

### Image pipeline

                 Uploaded Image
                       │
                       ▼
             Medical / Non-Medical
                       │
                       ▼
              Medical Image Type
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         CT           MRI          X-Ray
          │            │            │
          └────────────┼────────────┘
                       ▼
                Organ / Region
                       │
                       ▼
                Disease Class
                       │
                       ▼
              Prediction Result


### Supported model areas

- Brain MRI
- Chest CT
- Kidney CT
- Chest X-Ray
- Shoulder X-Ray
- Elbow X-Ray
- Hand X-Ray
- Humerus X-Ray
- Forearm X-Ray
- Medical image detection
- Medical image type classification


# 💬 Medical AI Chatbot

MedSource AI also contains a **Retrieval-Augmented Generation (RAG)** system.

Instead of asking the language model to answer entirely from its internal knowledge, the system first retrieves relevant medical documents and provides them as context to the LLM.

### RAG workflow

User Question
      │
      ▼
Text Embedding
      │
      ▼
Vector Search
      │
      ▼
Candidate Documents
      │
      ▼
Reranking
      │
      ▼
Top Relevant Context
      │
      ▼
Ollama LLM
      │
      ▼
AI Response

### Current RAG stack

- **ChromaDB** — vector database
- **all-MiniLM-L6-v2** — embedding model
- **MS MARCO MiniLM** — reranking
- **Ollama** — local LLM runtime
- **Qwen3 8B** — language model

The RAG implementation is located in:

medical_rag/


---

# 🛠️ Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn

### Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Joblib

### Deep Learning

- TensorFlow
- Keras
- CNN
- Transfer Learning

### Computer Vision

- OpenCV
- Image preprocessing
- CLAHE

### Generative AI / RAG

- Ollama
- Qwen3
- ChromaDB
- Sentence Transformers
- Reranking

### Frontend

- HTML5
- CSS3
- JavaScript

### Development

- Git
- GitHub
- Git LFS
- Docker

---

# 📁 Project Structure

```text
fast API/
│
├── app.py
├── requirments.txt
├── README.md
├── .gitignore
├── .gitattributes
│
├── Build_Models/
│   ├── training notebooks
│   └── model-building files
│
├── models/
│   ├── *.keras
│   └── *.joblib
│
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── disease prediction pages
│   ├── image diagnosis pages
│   └── chatbot pages
│
├── static/
│   └── css/
│
├── medical_rag/
│   ├── rag_api.py
│   ├── src/
│   ├── tests/
│   ├── README.md
│   └── requirements-lock-utf8.txt
│
└── Datasets/
```

> Local datasets, virtual environments, secrets, generated files, and the local Chroma database are excluded from version control where appropriate.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "fast API"
```

## 2. Create a Virtual Environment

```bash
python3 -m venv env
```

Activate it:

```bash
source env/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirments.txt
```

For the RAG module:

```bash
cd medical_rag
pip install -r requirements-lock-utf8.txt
cd ..
```

---

# 🔐 Environment Configuration

Sensitive configuration should never be committed to GitHub.

Create a local `.env` file when required:

```env
API_KEY=your_api_key
SECRET_KEY=your_secret_key
DATABASE_URL=your_database_url
```

Keep the real `.env` file private.

For GitHub, use an `.env.example` file containing only placeholder values.

---

# ▶️ Run the Application

Activate the virtual environment:

```bash
source env/bin/activate
```

Start FastAPI:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

# 📚 API Documentation

FastAPI automatically provides interactive API documentation.

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

# 🤖 GPU Support

The deep-learning components can use an NVIDIA GPU when a compatible CUDA/TensorFlow environment is configured.

Check GPU availability:

```python
import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
```

Example output:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

GPU acceleration is particularly useful during model training and image inference.

---

# 📦 Git LFS

The repository contains large trained machine-learning models.

Git LFS is used for files such as:

```text
*.keras
*.joblib
```

Install Git LFS:

```bash
git lfs install
```

After cloning the repository:

```bash
git lfs pull
```

Check LFS-managed files:

```bash
git lfs ls-files
```

---

# 🧪 Development

Create a feature branch:

```bash
git checkout -b feature-name
```

Check changes:

```bash
git status
```

Stage changes:

```bash
git add .
```

Commit changes:

```bash
git commit -m "Describe your changes"
```

Push the branch:

```bash
git push origin feature-name
```

---

# 🗃️ Dataset Policy

The project may use large medical datasets during training and evaluation.

Datasets are not included in the GitHub repository by default because of:

- Large file sizes
- Dataset licensing
- Storage requirements
- Privacy considerations
- Redistribution restrictions

Obtain datasets from their appropriate official sources and configure the local dataset paths required by the training scripts.

---

# 🔒 Security

Never commit:

```text
.env
API keys
Passwords
Database credentials
Secret keys
Private certificates
Personal medical information
```

The `.gitignore` file is configured to exclude common sensitive and generated files.

---

# ⚠️ Medical Disclaimer

**MedSource AI is an educational and research project.**

The predictions and information generated by this application:

- are not guaranteed to be medically accurate;
- are not a substitute for professional medical advice;
- should not be used to make medical decisions;
- should be evaluated by qualified healthcare professionals before any real-world clinical use.

---

# 🔮 Future Improvements

Planned improvements include:

- [ ] Improve disease prediction models
- [ ] Improve image classification accuracy
- [ ] Improve confidence calibration
- [ ] Expand medical image support
- [ ] Improve Grad-CAM visualization
- [ ] Integrate ECG prediction
- [ ] Improve RAG retrieval quality
- [ ] Improve chatbot response quality
- [ ] Add automated model evaluation
- [ ] Improve authentication and security
- [ ] Docker deployment
- [ ] Production API deployment
- [ ] Cloud deployment
- [ ] Better monitoring and logging

---

# 👨‍💻 Author

## Kripank MS

BCA Graduate with an interest in:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Generative AI
- RAG
- AI Agents
- Computer Vision
- FastAPI

---

# 📄 License

This project is intended primarily for educational and research purposes.

If this project is distributed publicly, add an appropriate license such as **MIT**, **Apache-2.0**, or another license that matches the project's requirements and the licenses of its dependencies and datasets.

---

# ⭐ Project Summary

**MedSource AI** combines traditional machine learning, deep learning, computer vision, and generative AI into a single medical-focused application.

```text
                 MEDSOURCE AI
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   ML Prediction   Image AI      RAG Chatbot
        │             │             │
        ▼             ▼             ▼
   Health Risk     CNN Models   Medical Docs
                      │             │
                      ▼             ▼
                  AI Analysis   AI Response
```

**Built with Python + FastAPI + TensorFlow + Machine Learning + RAG + Generative AI.**
