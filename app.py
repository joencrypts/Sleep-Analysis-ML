from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import pickle
import time
import google.generativeai as genai
import os
from datetime import datetime
import json

# Set API key immediately
os.environ['GEMINI_API_KEY'] = "AIzaSyAV3cNfrwJnbuo0VPNGas_5hdAqOn3TYYs"

# Configure page first - this must be the first Streamlit command
st.set_page_config(
    page_title="Sleep Health AI Analysis",
    page_icon="💤",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Configure Gemini AI
def configure_gemini():
    """Configure Gemini AI with API key"""
    try:
        # Get API key from environment (set at the top of the file)
        api_key = os.environ.get('GEMINI_API_KEY')
        
        if api_key and len(api_key) > 10:
            # Try the newer API syntax first
            try:
                from google import genai as google_genai
                client = google_genai.Client()
                return client
            except ImportError:
                # Fallback to older API
                genai.configure(api_key=api_key)
                # Try different model names (prioritize working ones)
                model_names = ['gemini-1.5-flash', 'gemini-2.5-flash', 'gemini-pro']
                for model_name in model_names:
                    try:
                        model = genai.GenerativeModel(model_name)
                        return model
                    except Exception:
                        continue
                return None
        else:
            st.warning("⚠️ Gemini API key not found or invalid.")
            return None
    except Exception as e:
        st.warning(f"⚠️ Gemini API configuration error: {str(e)[:100]}...")
        return None

# Initialize Gemini model
gemini_model = configure_gemini()

# Debug information (remove in production)
if st.sidebar.checkbox("🔧 Debug API Configuration"):
    st.sidebar.write("**API Key Status:**")
    try:
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            st.sidebar.write(f"✅ Found in environment: {api_key[:10]}...")
            st.sidebar.write(f"Key length: {len(api_key)} characters")
        else:
            st.sidebar.write("❌ Not found in environment")
            
        st.sidebar.write(f"**Gemini Model Status:** {'✅ Configured' if gemini_model else '❌ Not configured'}")
        
        if gemini_model:
            st.sidebar.success("🎉 Gemini AI is ready to use!")
        else:
            st.sidebar.error("❌ Gemini AI configuration failed")
    except Exception as e:
        st.sidebar.write(f"❌ Debug error: {e}")

# Manual report generation function
def generate_manual_report(data, stress_pred, sleep_score, health_score):
    """Generate a comprehensive manual health report"""
    
    # Risk assessment
    risk_level = "Low"
    if stress_pred >= 7 or sleep_score <= 4 or health_score <= 5:
        risk_level = "High"
    elif stress_pred >= 5 or sleep_score <= 6 or health_score <= 7:
        risk_level = "Medium"
    
    # Sleep analysis
    sleep_duration = data['Sleep Duration']
    sleep_quality = data['Quality of Sleep']
    
    if sleep_duration < 6:
        sleep_duration_advice = "⚠️ Your sleep duration is below recommended levels (7-9 hours)."
    elif sleep_duration > 9:
        sleep_duration_advice = "⚠️ Your sleep duration is above normal levels."
    else:
        sleep_duration_advice = "✅ Your sleep duration is within healthy range."
    
    if sleep_quality <= 4:
        sleep_quality_advice = "⚠️ Your sleep quality needs improvement."
    elif sleep_quality >= 7:
        sleep_quality_advice = "✅ You have good sleep quality."
    else:
        sleep_quality_advice = "📈 Your sleep quality is moderate with room for improvement."
    
    # Generate recommendations
    recommendations = []
    if data['Physical Activity Level'] < 30:
        recommendations.append("🏃‍♂️ Increase physical activity to at least 30 minutes daily")
    if data['Daily Steps'] < 6000:
        recommendations.append("👟 Aim for 8,000+ daily steps")
    if stress_pred >= 6:
        recommendations.append("🧘‍♀️ Practice stress management techniques")
    if sleep_duration < 7:
        recommendations.append("🌙 Prioritize getting 7-8 hours of sleep")
    if data['BMI Category'] in ['Overweight', 'Obese']:
        recommendations.append("🥗 Focus on balanced nutrition and weight management")
    
    # Display manual report
    st.markdown(f"""
    <div class="report-section">
        <h3>📊 Executive Summary</h3>
        <p>Based on your health profile, you have a <strong>{risk_level.lower()}</strong> risk level for stress-related health issues. 
        Your overall health score is <strong>{health_score:.1f}/10</strong>, indicating {'good' if health_score >= 7 else 'moderate' if health_score >= 5 else 'needs improvement'} health status.</p>
        
        <h3>🎯 Health Risk Assessment</h3>
        <p><strong>Risk Level:</strong> {risk_level}</p>
        <p><strong>Primary Concerns:</strong> {'High stress levels' if stress_pred >= 7 else 'Moderate stress' if stress_pred >= 5 else 'Low stress levels'}</p>
        
        <h3>🌙 Sleep Quality Analysis</h3>
        <p><strong>Sleep Duration:</strong> {sleep_duration_advice}</p>
        <p><strong>Sleep Quality:</strong> {sleep_quality_advice}</p>
        <p><strong>Sleep Score:</strong> {sleep_score:.1f}/10</p>
        
        <h3>💡 Lifestyle Recommendations</h3>
        <ul>
            {''.join([f'<li>{rec}</li>' for rec in recommendations])}
        </ul>
        
        <h3>🧘‍♀️ Stress Management Tips</h3>
        <ul>
            <li>Practice deep breathing exercises for 5-10 minutes daily</li>
            <li>Engage in regular physical activity</li>
            <li>Maintain a consistent sleep schedule</li>
            <li>Consider mindfulness or meditation practices</li>
            <li>Limit caffeine intake, especially in the afternoon</li>
        </ul>
        
        <h3>🎯 Long-term Health Goals</h3>
        <ul>
            <li>Achieve and maintain 7-8 hours of quality sleep nightly</li>
            <li>Reduce stress levels to below 5/10 through lifestyle changes</li>
            <li>Increase daily physical activity to 60+ minutes</li>
            <li>Maintain healthy blood pressure and heart rate</li>
            <li>Develop sustainable stress management routines</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #176397, #1D4665);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #00152b;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #176397;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .report-section {
        background: #000;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .recommendation-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>💤 Sleep Health AI Analysis Platform</h1>
    <p>Advanced Sleep Quality Prediction & Personalized Health Reports</p>
</div>
""", unsafe_allow_html=True)

# Navigation menu
selected = option_menu(
    menu_title=None,
    options=["🏠 Home", "📊 Analysis", "🔮 Prediction", "📋 AI Report", "📈 Insights"],
    icons=["house", "bar-chart", "cpu", "file-text", "lightbulb"],
    orientation="horizontal",
    default_index=0,
    styles={
        "nav-link-selected": {"background-color": "#176397"},
    }
)

# Load data and models
@st.cache_data
def load_data_and_models():
    """Load dataset and trained models"""
    try:
        df = pd.read_csv('.streamlit/Dataset/dataset.csv')
        df.drop(['Person ID', 'Sick'], axis=1, inplace=True)
        
        # Load models
        gender_le = joblib.load('.streamlit/Models/gender_encoder.pkl')
        occupation_le = joblib.load('.streamlit/Models/occupation_encoder.pkl')
        bmiCategory_le = joblib.load('.streamlit/Models/bmi_category_encoder.pkl')
        sleepDisorder_le = joblib.load('.streamlit/Models/sleep_disorder_encoder.pkl')
        scaler = joblib.load('.streamlit/Models/scaler.pkl')
        
        with open('.streamlit/Models/model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        return df, gender_le, occupation_le, bmiCategory_le, sleepDisorder_le, scaler, model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None

df, gender_le, occupation_le, bmiCategory_le, sleepDisorder_le, scaler, model = load_data_and_models()

if df is None:
    st.stop()

# Home Page
if selected == "🏠 Home":
    st.markdown("""
    ## Welcome to Sleep Health AI Analysis Platform
    
    This advanced platform combines machine learning with AI-powered insights to provide comprehensive sleep health analysis and personalized recommendations.
    
    ### 🌟 Key Features:
    - **Advanced Sleep Quality Prediction**: Predict stress levels and sleep quality using machine learning
    - **AI-Powered Health Reports**: Generate personalized reports using Google's Gemini AI
    - **Comprehensive Analysis**: Deep insights into sleep patterns, lifestyle factors, and health metrics
    - **Personalized Recommendations**: Get actionable advice tailored to your specific health profile
    
    ### 📊 Dataset Overview:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Age Range", f"{df['Age'].min()}-{df['Age'].max()}")
    with col4:
        st.metric("Model Accuracy", "96% R²")
    
    st.markdown("### 🎯 Model Performance:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>MAPE</h4>
            <h2>4%</h2>
            <p>Mean Absolute Percentage Error</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>RMSE</h4>
            <h2>0.37</h2>
            <p>Root Mean Square Error</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>R² Score</h4>
            <h2>96%</h2>
            <p>Coefficient of Determination</p>
        </div>
        """, unsafe_allow_html=True)

# Analysis Page
elif selected == "📊 Analysis":
    st.markdown("## 📊 Data Analysis & Insights")
    
    # Dataset overview
    st.markdown("### Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.markdown("### Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sleep Duration Distribution")
        fig = px.histogram(df, x='Sleep Duration', nbins=20, color_discrete_sequence=['#176397'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Stress Level vs Sleep Quality")
        fig = px.scatter(df, x='Quality of Sleep', y='Stress Level', 
                        color='Gender', size='Physical Activity Level',
                        color_discrete_sequence=['#176397', '#1D4665'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### Feature Correlation Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                   color_continuous_scale='RdBu_r')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Enhanced Prediction Page
elif selected == "🔮 Prediction":
    st.markdown("## 🔮 Advanced Sleep Health Prediction")
    
    st.markdown("""
    ### Provide Your Health Information
    Fill in the details below to get personalized sleep health predictions and recommendations.
    """)
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Personal Information")
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            age = st.slider("Age", 18, 80, 30, key="age")
            occupation = st.selectbox("Occupation", df['Occupation'].unique(), key="occupation")
            
            st.markdown("#### Sleep Information")
            sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0, 0.1, key="sleep_duration")
            sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 5, key="sleep_quality")
            
        with col2:
            st.markdown("#### Health Metrics")
            bmi_category = st.selectbox("BMI Category", df['BMI Category'].unique(), key="bmi")
            heart_rate = st.slider("Resting Heart Rate (bpm)", 50, 120, 70, key="heart_rate")
            sleep_disorder = st.selectbox("Sleep Disorder", df['Sleep Disorder'].unique(), key="sleep_disorder")
            
            st.markdown("#### Lifestyle Factors")
            physical_activity = st.slider("Physical Activity Level (minutes/day)", 0, 180, 60, key="activity")
            daily_steps = st.slider("Daily Steps", 1000, 20000, 8000, key="steps")
            bp_high = st.slider("Systolic Blood Pressure", 90, 180, 120, key="bp_high")
            bp_low = st.slider("Diastolic Blood Pressure", 50, 120, 80, key="bp_low")
        
        submitted = st.form_submit_button("🔮 Predict Sleep Health", use_container_width=True)
    
    if submitted:
        # Prepare input data
        input_data = {
            'Gender': gender,
            'Age': age,
            'Occupation': occupation,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': sleep_quality,
            'Physical Activity Level': physical_activity,
            'BMI Category': bmi_category,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Sleep Disorder': sleep_disorder,
            'BP High': bp_high,
            'BP Low': bp_low
        }
        
        # Convert to DataFrame
        prediction_df = pd.DataFrame(input_data, index=[0])
        
        # Encode categorical variables
        prediction_df['Gender'] = gender_le.transform(prediction_df['Gender'])
        prediction_df['Occupation'] = occupation_le.transform(prediction_df['Occupation'])
        prediction_df['BMI Category'] = bmiCategory_le.transform(prediction_df['BMI Category'])
        prediction_df['Sleep Disorder'] = sleepDisorder_le.transform(prediction_df['Sleep Disorder'])
        
        # Scale numerical features
        numerical_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                            'Heart Rate', 'Daily Steps', 'BP High', 'BP Low']
        prediction_df[numerical_features] = scaler.transform(prediction_df[numerical_features])
        
        # Make prediction
        with st.spinner('🔮 Analyzing your sleep health...'):
            stress_prediction = model.predict(prediction_df)[0]
            time.sleep(1)
        
        # Display prediction results
        st.markdown("### 🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Predicted Stress Level</h3>
                <h1>{stress_prediction:.1f}/10</h1>
                <p>Based on your health profile</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Calculate sleep quality score
            sleep_score = (sleep_quality * 2 + (8 - abs(sleep_duration - 7.5)) * 0.5) / 3
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Sleep Quality Score</h3>
                <h1>{sleep_score:.1f}/10</h1>
                <p>Comprehensive sleep assessment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calculate overall health score
            health_score = (10 - stress_prediction + sleep_score) / 2
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Overall Health Score</h3>
                <h1>{health_score:.1f}/10</h1>
                <p>Combined health assessment</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Store prediction data for AI report
        st.session_state.prediction_data = input_data
        st.session_state.stress_prediction = stress_prediction
        st.session_state.sleep_score = sleep_score
        st.session_state.health_score = health_score
        
        st.success("✅ Prediction completed! Go to 'AI Report' tab for detailed analysis.")

# AI Report Page
elif selected == "📋 AI Report":
    st.markdown("## 📋 AI-Powered Health Report")
    
    if 'prediction_data' not in st.session_state:
        st.warning("⚠️ Please complete a prediction first to generate your AI report.")
        st.info("Go to the 'Prediction' tab and submit your health information.")
    else:
        if gemini_model is None:
            st.error("❌ Gemini AI is not configured. Please set up the API key.")
            st.markdown("### Manual Health Analysis")
            
            # Manual analysis without AI
            data = st.session_state.prediction_data
            stress_pred = st.session_state.stress_prediction
            sleep_score = st.session_state.sleep_score
            health_score = st.session_state.health_score
            
            st.markdown(f"""
            <div class="report-section">
                <h3>📊 Your Health Profile Summary</h3>
                <p><strong>Age:</strong> {data['Age']} years</p>
                <p><strong>Gender:</strong> {data['Gender']}</p>
                <p><strong>Occupation:</strong> {data['Occupation']}</p>
                <p><strong>BMI Category:</strong> {data['BMI Category']}</p>
                <p><strong>Sleep Duration:</strong> {data['Sleep Duration']} hours</p>
                <p><strong>Sleep Quality:</strong> {data['Quality of Sleep']}/10</p>
                <p><strong>Physical Activity:</strong> {data['Physical Activity Level']} minutes/day</p>
                <p><strong>Daily Steps:</strong> {data['Daily Steps']:,}</p>
                <p><strong>Heart Rate:</strong> {data['Heart Rate']} bpm</p>
                <p><strong>Blood Pressure:</strong> {data['BP High']}/{data['BP Low']}</p>
                <p><strong>Sleep Disorder:</strong> {data['Sleep Disorder']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="report-section">
                <h3>🎯 Prediction Results</h3>
                <p><strong>Predicted Stress Level:</strong> {stress_pred:.1f}/10</p>
                <p><strong>Sleep Quality Score:</strong> {sleep_score:.1f}/10</p>
                <p><strong>Overall Health Score:</strong> {health_score:.1f}/10</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            # Generate AI report
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🤖 Generate AI Health Report", use_container_width=True):
                    with st.spinner("🤖 Generating your personalized AI health report..."):
                        try:
                            data = st.session_state.prediction_data
                            stress_pred = st.session_state.stress_prediction
                            sleep_score = st.session_state.sleep_score
                            health_score = st.session_state.health_score
                            
                            # Create prompt for Gemini
                            prompt = f"""
                            As a health AI specialist, analyze this sleep health data and provide a comprehensive, personalized health report:
                            
                            Patient Profile:
                            - Age: {data['Age']} years
                            - Gender: {data['Gender']}
                            - Occupation: {data['Occupation']}
                            - BMI Category: {data['BMI Category']}
                            - Sleep Duration: {data['Sleep Duration']} hours
                            - Sleep Quality: {data['Quality of Sleep']}/10
                            - Physical Activity: {data['Physical Activity Level']} minutes/day
                            - Daily Steps: {data['Daily Steps']:,}
                            - Heart Rate: {data['Heart Rate']} bpm
                            - Blood Pressure: {data['BP High']}/{data['BP Low']}
                            - Sleep Disorder: {data['Sleep Disorder']}
                            
                            Predictions:
                            - Predicted Stress Level: {stress_pred:.1f}/10
                            - Sleep Quality Score: {sleep_score:.1f}/10
                            - Overall Health Score: {health_score:.1f}/10
                            
                            Please provide:
                            1. Executive Summary (2-3 sentences)
                            2. Health Risk Assessment
                            3. Sleep Quality Analysis
                            4. Lifestyle Recommendations (5-7 specific, actionable items)
                            5. Stress Management Tips
                            6. Long-term Health Goals
                            
                            Format the response in clear sections with emojis and make it professional yet friendly.
                            """
                            
                            # Handle both new and old API syntax
                            if hasattr(gemini_model, 'models'):
                                # New API syntax
                                response = gemini_model.models.generate_content(
                                    model="gemini-2.5-flash",
                                    contents=prompt
                                )
                                ai_report = response.text
                            else:
                                # Old API syntax
                                response = gemini_model.generate_content(prompt)
                                ai_report = response.text
                            
                            st.session_state.ai_report = ai_report
                            st.success("✅ AI report generated successfully!")
                            
                        except Exception as e:
                            error_msg = str(e)
                            if "429" in error_msg or "RATE_LIMIT_EXCEEDED" in error_msg:
                                st.error("🚫 API rate limit exceeded. Please wait a minute and try again.")
                                st.info("💡 **Rate Limit Solutions:**\n- Wait 1-2 minutes before trying again\n- Consider upgrading your Gemini API plan\n- Use the manual analysis below for now")
                            elif "403" in error_msg or "PERMISSION_DENIED" in error_msg:
                                st.error("🔒 API access denied. Please check your API key permissions.")
                            elif "400" in error_msg or "INVALID_ARGUMENT" in error_msg:
                                st.error("❌ Invalid request. Please try again with different data.")
                            else:
                                st.error(f"❌ Error generating AI report: {error_msg[:200]}...")
                            
                            # Generate fallback manual report
                            st.markdown("### 📋 Manual Health Analysis (Fallback)")
                            generate_manual_report(data, stress_pred, sleep_score, health_score)
            
            with col2:
                if st.button("📋 Generate Manual Report", use_container_width=True):
                    data = st.session_state.prediction_data
                    stress_pred = st.session_state.stress_prediction
                    sleep_score = st.session_state.sleep_score
                    health_score = st.session_state.health_score
                    generate_manual_report(data, stress_pred, sleep_score, health_score)
            
            # Display AI report
            if 'ai_report' in st.session_state:
                st.markdown("### 🤖 Your Personalized AI Health Report")
                st.markdown(f"""
                <div class="report-section">
                    {st.session_state.ai_report.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
                
                # Download report
                report_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'patient_data': st.session_state.prediction_data,
                    'predictions': {
                        'stress_level': st.session_state.stress_prediction,
                        'sleep_score': st.session_state.sleep_score,
                        'health_score': st.session_state.health_score
                    },
                    'ai_report': st.session_state.ai_report
                }
                
                st.download_button(
                    label="📥 Download Health Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Insights Page
elif selected == "📈 Insights":
    st.markdown("## 📈 Health Insights & Trends")
    
    # Key insights from the dataset
    st.markdown("### 🔍 Key Insights from Our Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sleep Duration Analysis")
        avg_sleep = df['Sleep Duration'].mean()
        optimal_sleep = df[df['Quality of Sleep'] >= 7]['Sleep Duration'].mean()
        
        st.metric("Average Sleep Duration", f"{avg_sleep:.1f} hours")
        st.metric("Optimal Sleep Duration", f"{optimal_sleep:.1f} hours")
        
        # Sleep duration vs quality
        fig = px.scatter(df, x='Sleep Duration', y='Quality of Sleep', 
                        color='Stress Level', size='Physical Activity Level',
                        title="Sleep Duration vs Quality",
                        color_continuous_scale='RdYlBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Stress Level Patterns")
        avg_stress = df['Stress Level'].mean()
        high_stress_pct = (df['Stress Level'] >= 7).mean() * 100
        
        st.metric("Average Stress Level", f"{avg_stress:.1f}/10")
        st.metric("High Stress Population", f"{high_stress_pct:.1f}%")
        
        # Stress by occupation
        stress_by_occupation = df.groupby('Occupation')['Stress Level'].mean().sort_values(ascending=False)
        fig = px.bar(x=stress_by_occupation.index, y=stress_by_occupation.values,
                    title="Average Stress Level by Occupation")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations based on data
    st.markdown("### 💡 Data-Driven Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="recommendation-box">
            <h4>🌙 Sleep Optimization</h4>
            <p>• Aim for 7-8 hours of sleep</p>
            <p>• Maintain consistent sleep schedule</p>
            <p>• Create a relaxing bedtime routine</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="recommendation-box">
            <h4>🏃‍♂️ Physical Activity</h4>
            <p>• Target 60+ minutes daily activity</p>
            <p>• Aim for 8,000+ daily steps</p>
            <p>• Include both cardio and strength training</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="recommendation-box">
            <h4>🧘‍♀️ Stress Management</h4>
            <p>• Practice mindfulness and meditation</p>
            <p>• Maintain work-life balance</p>
            <p>• Seek professional help if needed</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>💤 Sleep Health AI Analysis Platform | Powered by Machine Learning & Gemini AI</p>
    <p>For educational and informational purposes only. Consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)