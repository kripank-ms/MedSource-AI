# 🏥 MedSource AI

> **AI-powered medical assistance, disease prediction, medical image analysis, and RAG platform**

MedSource AI is a full-stack medical AI application built with **Python, FastAPI, Machine Learning, Deep Learning, Computer Vision, Generative AI, and Retrieval-Augmented Generation (RAG)**.

The platform brings multiple AI capabilities together into a single web application, including:

- 🩺 Disease-risk prediction
- 🩻 Medical image analysis
- 🧠 Deep-learning-based medical image classification
- 💬 Medical AI chatbot using RAG
- 🔎 Medical document retrieval
- 👤 User authentication and account management
- 🗄️ MongoDB-based user information storage

> **Medical Disclaimer:** MedSource AI is an educational and research project. Predictions and AI-generated information are not a substitute for professional medical advice, diagnosis, or treatment.

---

# ✨ Features

| Feature | Technology |
|---|---|
| 🩺 Disease prediction | Scikit-learn / Joblib |
| 🩻 Medical image analysis | TensorFlow / Keras |
| 🧠 Brain MRI analysis | Deep Learning |
| 🫁 Chest CT analysis | Deep Learning |
| 🫘 Kidney CT analysis | Deep Learning |
| 🩻 X-Ray classification | Deep Learning |
| 💬 Medical AI chatbot | RAG + Ollama |
| 🔎 Medical document retrieval | ChromaDB |
| 🧮 Text embeddings | Sentence Transformers |
| 🔐 Authentication | FastAPI |
| 👤 User information | MongoDB |
| 🌐 Backend API | FastAPI |
| 🎨 Web interface | HTML + CSS + JavaScript |
| 📦 Large model storage | Git LFS |

---

# 🧠 System Architecture

```text
                         ┌─────────────────────┐
                         │        User         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    MedSource AI     │
                         │     Web Interface   │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │ Disease        │ │ Medical Image  │ │ Medical        │
        │ Prediction     │ │ Analysis       │ │ Chatbot        │
        │ ML / Joblib    │ │ CNN / Keras    │ │ RAG + Ollama   │
        └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
                │                  │                  │
                ▼                  ▼                  ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │ Risk           │ │ Image          │ │ ChromaDB       │
        │ Prediction     │ │ Classification │ │ Vector Search  │
        └────────────────┘ └────────────────┘ └───────┬────────┘
                                                       │
                                                       ▼
                                                ┌──────────────┐
                                                │ Reranker     │
                                                └──────┬───────┘
                                                       │
                                                       ▼
                                                ┌──────────────┐
                                                │ Ollama       │
                                                │ Qwen3 8B     │
                                                └──────────────┘


                         User Account
                              │
                              ▼
                       ┌─────────────┐
                       │  MongoDB    │
                       │ User Data   │
                       └─────────────┘
```

---

# 🩺 Disease Prediction

MedSource AI contains machine-learning models for different health-related prediction tasks.

Current model areas include:

- ❤️ Heart Disease
- 🫁 COPD
- 🧬 HIV Risk
- 🧠 Stroke
- 🫘 CKD / Kidney Disease
- 🦠 Tuberculosis

### Prediction workflow

```text
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
```

The prediction models are intended for **educational and research purposes** and should not be used as standalone medical diagnostic systems.

---

# 🩻 Medical Image Analysis

MedSource AI uses deep-learning models to classify and analyze medical images.

### Image pipeline

```text
Uploaded Image
      │
      ▼
Medical / Non-Medical
      │
      ▼
Medical Image Type
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
     CT             MRI           X-Ray
      │              │              │
      └──────────────┼──────────────┘
                     ▼
              Organ / Region
                     │
                     ▼
                Disease Class
                     │
                     ▼
              Prediction Result
```

### Supported image areas

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

---

# 💬 Medical AI Chatbot

MedSource AI includes a **Retrieval-Augmented Generation (RAG)** system for answering medical questions using retrieved reference documents.

Instead of relying only on the language model's internal knowledge, the system retrieves relevant medical information and provides it as context to the LLM.

### RAG workflow

```text
User Question
      │
      ▼
Question / Intent Processing
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
Relevant Context
      │
      ▼
Ollama LLM
      │
      ▼
Grounded AI Response
      │
      ▼
Sources
```

### RAG stack

- **ChromaDB** — vector database
- **all-MiniLM-L6-v2** — embedding model
- **MS MARCO MiniLM** — reranking model
- **Ollama** — local LLM runtime
- **Qwen3 8B** — language model
- **Sentence Transformers** — text embeddings

Detailed RAG documentation is available in:

```text
medical_rag/README.md
```

---

# 👤 User Authentication & MongoDB

MedSource AI includes user authentication and account-management functionality.

**MongoDB is used to store user/account information.**

The application can use MongoDB for information associated with registered users and account management.

The MongoDB database is separate from the RAG vector database:

```text
                 MedSource AI
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
      MongoDB                  ChromaDB
          │                       │
          ▼                       ▼
   User Information          RAG Documents
   & Accounts                & Embeddings
```

### Important security rule

MongoDB connection strings and credentials must **never** be committed to GitHub.

Use environment variables or a local `.env` file instead.

Example:

```env
MONGODB_URL=your_mongodb_connection_string
```

The real value must remain private.

---

# 🛠️ Technology Stack

## Backend

- Python
- FastAPI
- Uvicorn

## Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Joblib

## Deep Learning

- TensorFlow
- Keras
- CNN
- Transfer Learning

## Computer Vision

- OpenCV
- Image preprocessing
- CLAHE

## Generative AI / RAG

- Ollama
- Qwen3
- ChromaDB
- Sentence Transformers
- CrossEncoder / Reranking

## Database

- MongoDB — user/account information
- ChromaDB — RAG vector storage

## Frontend

- HTML5
- CSS3
- JavaScript

## Development

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

> Datasets, virtual environments, secrets, generated files, and the local ChromaDB database are excluded from version control where appropriate.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/kripank-ms/MedSource-AI.git
cd MedSource-AI
```

## 2. Create a virtual environment

```bash
python3 -m venv env
```

Activate it:

```bash
source env/bin/activate
```

## 3. Install dependencies

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

Sensitive configuration must never be committed to GitHub.

Create a local `.env` file when required:

```env
MONGODB_URL=your_mongodb_connection_string
SECRET_KEY=your_secret_key
```

Keep the real `.env` file private.

For GitHub, use an `.env.example` file containing only placeholder values.

Example:

```env
MONGODB_URL=your_mongodb_connection_string
SECRET_KEY=your_secret_key
```

Never place real passwords, API keys, database credentials, or secret keys in `.env.example`.

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

Example:

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

After cloning:

```bash
git lfs pull
```

Check LFS-managed files:

```bash
git lfs ls-files
```

---

# 🗃️ Dataset Policy

Medical datasets used for training and evaluation are **not included in the GitHub repository**.

Reasons include:

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
ChromaDB data
Virtual environments
```

The `.gitignore` file is configured to exclude common sensitive and generated files.

---

# ⚠️ Medical Disclaimer

**MedSource AI is an educational and research project.**

The predictions and information generated by this application:

- are not guaranteed to be medically accurate;
- are not a substitute for professional medical advice;
- should not be used to make medical decisions;
- should be evaluated by qualified healthcare professionals before real-world clinical use.

Do not use this project as a replacement for a doctor, pharmacist, or other qualified healthcare professional.

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

This project is primarily intended for educational and research purposes.

If this project is distributed publicly, add an appropriate open-source license after checking the licenses of the dependencies, datasets, and other third-party components.

---

# ⭐ Project Summary

**MedSource AI** combines traditional machine learning, deep learning, computer vision, and generative AI into a single medical-focused application.

```text
                    MEDSOURCE AI
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
     ML Prediction    Image AI      RAG Chatbot
          │              │              │
          ▼              ▼              ▼
      Health Risk     CNN Models   Medical Docs
          │              │              │
          │              ▼              ▼
          │          AI Analysis    AI Response
          │
          ▼
     User Account
          │
          ▼
       MongoDB
```

**Built with Python + FastAPI + TensorFlow + Machine Learning + Computer Vision + RAG + Generative AI + MongoDB.**
