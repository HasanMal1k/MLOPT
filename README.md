#  MLOPT - Automated Machine Learning Platform

<div align="center">
  <img src="https://github.com/user-attachments/assets/d1d3da5a-3b56-4e08-a2f7-69e1980f1cac" alt="MLOPT Logo" width="1000" />

  
  **Transform your data into intelligent models with zero coding required**
  
  [![Next.js](https://img.shields.io/badge/Next.js-14.2-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![PyCaret](https://img.shields.io/badge/PyCaret-3.0-blue?style=for-the-badge)](https://pycaret.org/)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=for-the-badge&logo=typescript)](https://www.typescriptlang.org/)
  [![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://python.org/)
  [![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](https://docker.com/)
  [![Azure ML](https://img.shields.io/badge/Azure_ML-Integrated-0089D6?style=for-the-badge&logo=microsoft-azure)](https://azure.microsoft.com/en-us/services/machine-learning/)
</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Video Tutorials](#-video-tutorials)
- [Project Structure](#-project-structure)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🎯 Overview

**MLOPT** is a comprehensive **Final Year Project** that automates the complete data science pipeline. Built in collaboration with [@jawad-idrees](https://github.com/jawad-idrees), this platform enables users to go from raw data to trained machine learning models with just a few clicks.

### The Problem We Solve

Traditional machine learning workflows require:
- 📚 Extensive programming knowledge
- ⏰ Hours of manual data preprocessing
- 🔧 Complex feature engineering
- 📊 Algorithm selection expertise
- 🚀 Infrastructure for model deployment

### Our Solution

**MLOPT** automates all of this:
1. **Drag & Drop** your data files
2. **Auto-preprocessing** cleans and prepares your data
3. **Smart transformations** optimize features automatically
4. **Multiple models** are trained and compared
5. **One-click deployment** to Azure ML

---

## ✨ Features

### 📁 Data Management
- **Multi-format Support**: CSV, Excel (.xlsx, .xls), ZIP archives
- **Kaggle Integration**: Import datasets directly from Kaggle URLs
- **Data Preview**: Interactive preview with statistics
- **Quality Scoring**: Automatic data quality assessment (0-100)

### 🔄 Auto Preprocessing
- **Missing Value Handling**: Smart imputation strategies
- **Outlier Detection**: IQR and Z-score methods
- **Data Type Inference**: Automatic column type detection
- **Duplicate Removal**: Intelligent deduplication
- **Custom Cleaning**: Manual control when needed

### 🔧 Transformations
- **Encoding**: Label encoding, One-hot encoding, Target encoding
- **Scaling**: Standard, MinMax, Robust scaling
- **Feature Engineering**: Polynomial features, Binning
- **Dimensionality Reduction**: PCA, Feature selection

### 🧠 Auto ML Training
| Task Type | Supported Algorithms |
|-----------|---------------------|
| **Classification** | Random Forest, XGBoost, LightGBM, SVM, KNN, Logistic Regression, Decision Tree, Gradient Boosting, AdaBoost, MLP, Naive Bayes, QDA, Ridge, Extra Trees |
| **Regression** | Random Forest, XGBoost, LightGBM, Linear Regression, Ridge, Lasso, ElasticNet, SVR, KNN, Decision Tree, Gradient Boosting, AdaBoost, MLP, Huber |
| **Time Series** | Prophet, ARIMA, Auto ARIMA, Exponential Smoothing, Theta, TBATS |

### 📊 Exploratory Data Analysis (EDA)
- **Auto-generated Reports**: Comprehensive data profiling
- **Interactive Charts**: Custom chart builder
- **Statistical Analysis**: Correlation, distribution, outliers
- **Export Options**: PDF reports

### 🚀 Model Deployment
- **Azure ML Integration**: One-click deployment
- **Endpoint Management**: Monitor deployed models
- **Model Download**: Export trained models (.pkl)
- **Prediction API**: Real-time inference

### 🔐 User Management
- **Authentication**: Google OAuth, Email/Password
- **Subscription Plans**: Free, Pro, Enterprise tiers
- **Usage Tracking**: Monitor API calls and storage

---

## 🛠 Tech Stack

### Frontend
| Technology | Purpose |
|------------|---------|
| **Next.js 14** | React framework with App Router |
| **TypeScript** | Type-safe JavaScript |
| **Tailwind CSS** | Utility-first styling |
| **shadcn/ui** | Beautiful UI components |
| **Recharts** | Data visualization |
| **React Hook Form** | Form management |
| **Zod** | Schema validation |

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance Python API |
| **PyCaret 3.0** | AutoML library |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Machine learning |
| **YData Profiling** | EDA report generation |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| **Supabase** | Database & Authentication |
| **Azure ML** | Model deployment |
| **Docker** | Containerization |
| **PostgreSQL** | Data storage |

### DevOps
| Technology | Purpose |
|------------|---------|
| **Docker Compose** | Multi-container orchestration |
| **GitHub Actions** | CI/CD pipelines |
| **Vercel** | Frontend hosting |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MLOPT Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │   Frontend   │────▶│   Backend    │────▶│   Azure ML   │    │
│  │   (Next.js)  │◀────│   (FastAPI)  │◀────│  (Deployment)│    │
│  └──────────────┘     └──────────────┘     └──────────────┘     │
│         │                    │                                  │
│         │                    │                                  │
│         ▼                    ▼                                  │
│  ┌──────────────┐     ┌──────────────┐                          │
│  │   Supabase   │     │   PyCaret    │                          │
│  │  (Auth + DB) │     │  (Auto ML)   │                          │
│  └──────────────┘     └──────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Upload → Validation → Preprocessing → Transformation → Training → Evaluation → Deployment
     │            │              │               │              │           │            │
     ▼            ▼              ▼               ▼              ▼           ▼            ▼
  CSV/Excel   Format Check   Clean Data    Encode/Scale    AutoML     Metrics     Azure ML
  ZIP/Kaggle  Size Check     Impute        Feature Eng.   Compare    Download    Endpoint
```

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+ 
- **Python** 3.10+
- **Docker** & Docker Compose
- **Supabase** Account
- **Azure** Account (for deployment features)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/MLOPT.git
cd MLOPT
```

2. **Setup Frontend**
```bash
cd client
npm install
cp .env.example .env
# Edit .env with your Supabase credentials
```

3. **Setup Backend (Docker)**
```bash
cd server
docker build -t mlopt:v3 .
```

4. **Configure Environment Variables**

**Frontend (.env)**
```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

**Backend (Docker run)**
```env
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
AZURE_TENANT_ID=your_tenant_id
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_RESOURCE_GROUP=your_resource_group
AZURE_WORKSPACE_NAME=your_workspace_name
```

5. **Run the Application**

**Start Backend:**
```powershell
cd server
docker run -d --rm -p 8000:8000 `
  --name mlopt `
  -v ${PWD}:/app `
  -e SUPABASE_URL="your_supabase_url" `
  -e SUPABASE_SERVICE_KEY="your_service_key" `
  -e AZURE_TENANT_ID="your_tenant_id" `
  -e AZURE_CLIENT_ID="your_client_id" `
  -e AZURE_CLIENT_SECRET="your_client_secret" `
  -e AZURE_SUBSCRIPTION_ID="your_subscription_id" `
  -e AZURE_RESOURCE_GROUP="your_resource_group" `
  -e AZURE_WORKSPACE_NAME="your_workspace_name" `
  mlopt:v3
```

**Start Frontend:**
```bash
cd client
npm run dev
```

6. **Access the Application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 📖 Usage Guide

### Step 1: Upload Your Data

1. Navigate to **Data Upload**
2. Drag & drop your CSV/Excel file, or
3. Paste a Kaggle dataset URL
4. Click **Upload**

### Step 2: Preprocessing

1. Review auto-detected data quality
2. Apply auto-preprocessing or customize:
   - Handle missing values
   - Remove duplicates
   - Fix outliers
3. Name your cleaned dataset

### Step 3: Transformations

1. Select encoding method for categorical columns
2. Choose scaling for numerical columns
3. Apply feature engineering if needed
4. Preview transformations

### Step 4: Train Models

1. Go to **ML Blueprints**
2. Select your preprocessed dataset
3. Choose target column
4. Configure training parameters
5. Click **Start Training**
6. Watch real-time training progress

### Step 5: Evaluate & Deploy

1. Review model leaderboard
2. Compare metrics (Accuracy, F1, AUC, etc.)
3. Download best model (.pkl)
4. Deploy to Azure ML (optional)

---

## 📚 API Documentation

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/preprocess/` | Auto-preprocess uploaded data |
| `POST` | `/custom-clean/` | Apply custom cleaning rules |
| `POST` | `/configure-training/` | Start ML training job |
| `GET` | `/training-stream/{id}` | Stream training progress (SSE) |
| `POST` | `/predict/` | Make predictions |
| `GET` | `/download-model/{id}/{name}` | Download trained model |

### EDA Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate-eda-report` | Generate profiling report |
| `POST` | `/analyze-quality/` | Get data quality score |

### Deployment Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/deployments/deploy` | Deploy model to Azure |
| `GET` | `/deployments/list` | List all deployments |
| `DELETE` | `/deployments/{id}` | Delete deployment |

Full API documentation available at `/docs` when running the backend.

---

## 🎥 Video Tutorials

| Topic | Video Link |
|-------|------------|
| Landing Page & Login | [Watch](https://www.youtube.com/watch?v=0VqBUZTtYYs) |
| Auto & Custom Preprocessing | [Watch](https://youtu.be/Y14MTUuC3y4) |
| Transformations Activity | [Watch](https://www.youtube.com/watch?v=znlODwYKlrI) |
| Auto Training - Classification | [Watch](https://www.youtube.com/watch?v=vTEQ2c_OuPY) |
| Auto Training - Regression | [Watch](https://www.youtube.com/watch?v=fBgMuqsSgB0) |
| Auto Training - Time Series | [Watch](https://www.youtube.com/watch?v=tNdKW_StDAQ) |
| Blueprint Design - Classification | [Watch](https://www.youtube.com/watch?v=9ICbWWXQIF4) |
| Blueprint Design - Regression | [Watch](https://www.youtube.com/watch?v=1KXqaFrR6HY) |
| EDA & Chart Builder | [Watch](https://www.youtube.com/watch?v=mAPa38sAR0I) |
| Time Series Preprocessing | [Watch](https://www.youtube.com/watch?v=tw88arY6B1o) |

---

## 📁 Project Structure

```
MLOPT/
├── client/                    # Next.js Frontend
│   ├── app/                   # App Router pages
│   │   ├── dashboard/         # Main dashboard
│   │   │   ├── blueprints/    # ML training
│   │   │   ├── data-upload/   # File upload
│   │   │   ├── datasets/      # Data management
│   │   │   ├── deployments/   # Azure deployment
│   │   │   ├── eda/           # Exploratory analysis
│   │   │   ├── models/        # Trained models
│   │   │   └── tutorials/     # Video tutorials
│   │   ├── auth/              # Authentication
│   │   └── api/               # API routes
│   ├── components/            # React components
│   │   ├── ui/                # shadcn/ui components
│   │   └── upload/            # Upload wizard components
│   ├── lib/                   # Utilities
│   └── utils/                 # Helper functions
│
├── server/                    # FastAPI Backend
│   ├── main.py                # Main application
│   ├── ml_training.py         # ML training logic
│   ├── data_preprocessing.py  # Preprocessing functions
│   ├── data_quality.py        # Quality scoring
│   ├── azure_deployment.py    # Azure ML integration
│   ├── model_management.py    # Model CRUD operations
│   ├── Dockerfile             # Container configuration
│   └── requirements.txt       # Python dependencies
│
├── database/                  # SQL migrations
│   └── schema.sql             # Database schema
│
└── README.md                  # This file
```

---

## 🗄 Database Schema

### Core Tables

| Table | Description |
|-------|-------------|
| `users` | User accounts (via Supabase Auth) |
| `files` | Uploaded datasets metadata |
| `trained_models` | Trained ML models |
| `deployments` | Azure ML deployments |
| `subscriptions` | User subscription plans |

---

## 🤝 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/HasanMal1k/">
        <img src="https://github.com/HasanMal1k.png" width="100px;" alt=""/>
        <br />
        <sub><b>Muhammad Hasan</b></sub>
      </a>
      <br />
      <sub>Full Stack Development</sub>
    </td>
    <td align="center">
      <a href="https://github.com/jawad-idrees">
        <img src="https://github.com/jawad-idrees.png" width="100px;" alt=""/>
        <br />
        <sub><b>Jawad Idrees</b></sub>
      </a>
      <br />
      <sub>ML Pipeline & Backend</sub>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [PyCaret](https://pycaret.org/) - AutoML library
- [Supabase](https://supabase.com/) - Backend as a Service
- [shadcn/ui](https://ui.shadcn.com/) - Beautiful UI components
- [Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/) - Model deployment

---

<div align="center">
  <p>Built with ❤️ as a Final Year Project</p>
  <p>
    <a href="https://github.com/yourusername/MLOPT/issues">Report Bug</a>
    ·
    <a href="https://github.com/yourusername/MLOPT/issues">Request Feature</a>
  </p>
</div>
