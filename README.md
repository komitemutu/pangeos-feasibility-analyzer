# 🛥️ Pangeos Feasibility Analyzer v2.0

Advanced AI-powered yacht project feasibility analysis system with Qwen 3 multi-agent intelligence.

## ✨ Features

### 🤖 Multi-Agent AI System
- **Engineering Agent**: Qwen-2.5-72B for structural & propulsion analysis
- **Financial Agent**: Qwen-2-72B for cost modeling & ROI calculations  
- **Environmental Agent**: Qwen-1.5-110B for sustainability assessment
- **Logistics Agent**: Qwen-2.5-7B for operational planning

### 📊 Analysis Capabilities
- **Quick Analysis**: Rapid feasibility assessment
- **Deep Multi-Agent Analysis**: Comprehensive AI-powered evaluation
- **Comparative Analysis**: Scenario comparison (Conservative/Optimistic/Ambitious)
- **Scenario Planning**: Monte Carlo simulations with statistical analysis

### 🚀 Technology Stack
- **Frontend**: Streamlit with custom CSS styling
- **AI Models**: Qwen 3 family via OpenRouter API
- **Visualization**: Plotly interactive charts
- **Data Processing**: Pandas & NumPy
- **Deployment**: Vercel with Python 3.9

## 🛠️ Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/komitemutu/pangeos-feasibility-analyzer.git
cd pangeos-feasibility-analyzer
```

### 2. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

### 3. Deploy to Vercel
```bash
# Connect to Vercel
vercel

# Or push to GitHub and connect via Vercel dashboard
# Vercel will auto-deploy using vercel.json configuration
```

## 📁 Project Structure
```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies  
├── runtime.txt           # Python version for deployment
├── vercel.json           # Vercel deployment config
├── .streamlit/
│   └── secrets.toml      # Configuration secrets
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🔧 Configuration

### Free AI Models (No API Key Required)
The application uses free Qwen models from OpenRouter:
- `qwen/qwen-2.5-72b-instruct` - Primary engineering analysis
- `qwen/qwen-2-72b-instruct` - Financial modeling
- `qwen/qwen-1.5-110b-chat` - Environmental assessment  
- `qwen/qwen-2.5-7b-instruct` - Logistics planning

### Optional: Enhanced Rate Limits
For higher rate limits, add OpenRouter API key to `.streamlit/secrets.toml`:
