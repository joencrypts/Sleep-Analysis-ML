# 💤 Sleep Health AI Analysis Platform

## 🌟 Advanced Sleep Quality Prediction & Personalized Health Reports

This project combines machine learning with AI-powered insights to provide comprehensive sleep health analysis and personalized recommendations using Google's Gemini AI.

### 🚀 New Features

- **🤖 AI-Powered Health Reports**: Generate personalized, comprehensive health reports using Google's Gemini AI
- **🔮 Advanced Prediction Interface**: Enhanced user-friendly interface for sleep health predictions
- **📊 Comprehensive Data Analysis**: Deep insights into sleep patterns, lifestyle factors, and health metrics
- **📈 Interactive Visualizations**: Dynamic charts and graphs for better data understanding
- **📋 Personalized Recommendations**: Get actionable advice tailored to your specific health profile
- **📥 Report Export**: Download your health reports in JSON format

### 🎯 Project Overview

This project includes the analysis of sleep health and lifestyle dataset and an application for predicting stress levels using machine learning. The dataset consists of 400 rows and 13 columns, encompassing various demographic, health, and lifestyle variables.

### 🎯 Project Objectives

The main objectives of the project are to analyze and visualize the data related to health, lifestyle, and demographic factors, derive actionable insights from the visualizations, and predict stress levels of individuals using machine learning techniques.

### 🌟 Project Features

- **Sleep health metrics analysis**: Explore factors related to sleep duration, quality, and regularity.
- **Lifestyle factors analysis**: Investigate physical activity levels, stress levels, and BMI categories.
- **Cardiovascular health analysis**: Examine blood pressure and resting heart rate measurements.
- **Sleep disorder analysis**: Determine the presence of sleep disorders such as insomnia and sleep apnea.
- **AI-powered insights**: Generate personalized health reports using Google's Gemini AI
- **Interactive prediction interface**: User-friendly form for health data input
- **Comprehensive visualizations**: Dynamic charts and analysis tools

### 🤖 AI Integration

The platform now integrates with Google's Gemini AI to provide:
- Personalized health reports
- Detailed analysis of your health profile
- Actionable recommendations
- Risk assessments
- Long-term health goals

**Setup Instructions:**
1. Get a Gemini API key from: https://makersuite.google.com/app/apikey
2. Create a `.streamlit/secrets.toml` file in your project root
3. Add your API key: `GEMINI_API_KEY = "your_api_key_here"`
4. Or set environment variable: `export GEMINI_API_KEY="your_api_key_here"`

**Note:** The app works without Gemini AI, but AI report generation will be disabled.

### 📊 Model Performance

- **MAPE**: 4% (Mean Absolute Percentage Error)
- **RMSE**: 0.37 (Root Mean Square Error)  
- **R² Score**: 96% (Coefficient of Determination)

### 🛠️ Technologies Used

- **Data Visualization**: Tableau, Plotly for interactive charts
- **Data Analysis**: Python, Jupyter Notebook, Pandas, NumPy
- **Machine Learning**: scikit-learn, Linear Regression
- **AI Integration**: Google Generative AI (Gemini)
- **Application Development**: Streamlit with enhanced UI
- **Data Processing**: Joblib for model serialization

### 📋 Dataset Information

The dataset consists of 400 rows and 13 columns, containing various variables related to sleep health and lifestyle:

1. **Person ID**: Unique identifier for each individual
2. **Gender**: Gender of the person (Male/Female)
3. **Age**: Age of the person in years
4. **Occupation**: Person's occupation or profession
5. **Sleep Duration (hours)**: Number of hours slept by the person in a day
6. **Quality of Sleep (scale: 1-10)**: Subjective evaluation of sleep quality
7. **Physical Activity Level (minutes/day)**: Daily physical activity minutes
8. **Stress Level (scale: 1-10)**: Subjective evaluation of stress level
9. **BMI Category**: BMI category (Underweight, Normal, Overweight, Obese)
10. **Blood Pressure (systolic/diastolic)**: Blood pressure measurements
11. **Resting Heart Rate (bpm)**: Resting heart rate in beats per minute
12. **Daily Steps**: Number of steps taken per day
13. **Sleep Disorder**: Presence of sleep disorders (None, Insomnia, Sleep Apnea)

### 🚀 Installation and Usage

#### Option 1: Quick Setup
```bash
# Clone the repository
git clone https://github.com/joencrypts/Sleep-Analysis-ML.git
cd Sleep-Analysis-ML

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

#### Option 2: Windows Batch Installer
Just double-click on the `install.bat` file in the main directory.

#### Option 3: Manual Installation
```bash
pip install streamlit streamlit_option_menu pandas numpy joblib plotly scikit-learn google-generativeai
```

### 🌐 Online Access

- **📊 Dashboard**: [Interactive Tableau Dashboard](https://public.tableau.com/app/profile/ramazan.erduran1816/viz/StressLevelHealth/Overview)
- **🌐 Web Application**: [Streamlit Cloud Deployment](https://sleep-health-ml-project.streamlit.app/)

### 📱 Application Features

#### 🏠 Home Page
- Project overview and key metrics
- Model performance indicators
- Dataset statistics

#### 📊 Analysis Page
- Interactive data visualizations
- Statistical summaries
- Correlation analysis
- Feature distribution charts

#### 🔮 Prediction Page
- User-friendly input form
- Real-time predictions
- Multiple health scores
- Comprehensive health assessment

#### 📋 AI Report Page
- AI-powered health reports
- Personalized recommendations
- Risk assessments
- Report export functionality

#### 📈 Insights Page
- Data-driven insights
- Health trend analysis
- Personalized recommendations
- Best practices

### 🔧 Configuration

#### Gemini AI Setup
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create `.streamlit/secrets.toml`:
```toml
[secrets]
GEMINI_API_KEY = "your_api_key_here"
```
3. Or set environment variable:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 📁 Project Structure

```
Sleep-Analysis-ML/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── install.bat                     # Windows installer script
├── README.md                       # Project documentation
├── Datasets/
│   ├── raw-dataset.csv            # Original dataset
│   └── cleaned-dataset.csv        # Processed dataset
├── Models/
│   └── linear-regression.pkl      # Trained model
├── Notebooks/
│   ├── exploring.ipynb            # Data exploration
│   ├── machine-learning-playground.ipynb  # ML model training
│   └── visualizationPlayground.ipynb      # Data visualization
├── .streamlit/
│   ├── app.py                     # Original app (backup)
│   ├── Dataset/
│   │   └── dataset.csv           # App dataset
│   ├── Models/                    # Model files
│   └── secrets.toml.example       # Configuration template
├── Imgs/                          # Application images
└── Tableau/                       # Tableau dashboard files
```

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### ⚠️ Disclaimer

This application is for educational and informational purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

### 👨‍💻 Author

**Ramazan ERDURAN**

---

<div align="center">
    <p>💤 Sleep Health AI Analysis Platform | Powered by Machine Learning & Gemini AI</p>
    <p>For educational and informational purposes only. Consult healthcare professionals for medical advice.</p>
</div>
