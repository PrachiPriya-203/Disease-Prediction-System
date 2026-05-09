import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Sympto Medics | AI Disease Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Minimal Light Design ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #111827;
        background-color: #F9FAFB;
    }
    
    .stApp {
        background-color: #F9FAFB;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #111827;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #6B7280;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .custom-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
        border: 1px solid #F3F4F6;
    }
    
    /* Metric styling */
    div[data-testid="metric-container"] {
        background-color: #F9FAFB;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02);
        border: 1px solid #E5E7EB;
        text-align: center;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #4F46E5;
        font-weight: 600;
    }
    
    /* Button styling */
    div.stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2);
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 8px -1px rgba(79, 70, 229, 0.3);
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
        color: white;
    }
    
    /* Multiselect styling */
    .stMultiSelect div[data-baseweb="select"] {
        border-radius: 8px;
    }
    
    hr {
        border-color: #E5E7EB;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model & Features ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/random_forest_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, feature_names
    except FileNotFoundError:
        return None, None

model, feature_names = load_assets()

# --- Main App ---
def main():
    st.markdown("<h1 class='main-header'>🩺 Sympto Medics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Based Disease Prediction System</p>", unsafe_allow_html=True)
    
    if model is None or feature_names is None:
        st.error("⚠️ Model or Feature Names not found. Please run the training script (`train_models.py`) first.")
        return

    # Clean up feature names for display
    display_features = [f.replace('_', ' ').title() for f in feature_names]
    
    # Input Section Card
    with st.container(border=True):
        st.markdown("### Select Symptoms")
        
        # Multi-select for symptoms
        selected_symptoms_display = st.multiselect(
            "Search and choose the symptoms you are experiencing:",
            options=display_features,
            default=[]
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Analyze & Predict")
    
    # Result Section
    if predict_btn:
        if len(selected_symptoms_display) == 0:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analyzing data..."):
                # Map selected display names back to original feature names
                selected_features = [feature_names[display_features.index(sym)] for sym in selected_symptoms_display]
                
                # Create input array
                input_data = np.zeros(len(feature_names))
                for i, feature in enumerate(feature_names):
                    if feature in selected_features:
                        input_data[i] = 1
                        
                input_df = pd.DataFrame([input_data], columns=feature_names)
                
                # Predict
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                confidence = np.max(probabilities) * 100
                
                # Top Possibilities
                top_indices = np.argsort(probabilities)[::-1][:3]
                top_classes = model.classes_[top_indices]
                top_probs = probabilities[top_indices] * 100
                
                # Render Results Card
                with st.container(border=True):
                    st.markdown("### Prediction Results")
                    
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(label="Primary Diagnosis", value=prediction)
                    with m2:
                        st.metric(label="Confidence Score", value=f"{confidence:.1f}%")
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### Possible Conditions")
                    for cls, prob in zip(top_classes, top_probs):
                        if prob > 0:
                            st.progress(prob/100, text=f"{cls} ({prob:.1f}%)")
                    
                    st.markdown("---")
                    st.markdown("### Explainable AI (SHAP)")
                    st.write("Symptom impact on the prediction:")
                    
                    # SHAP Analysis
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(input_df)
                        
                        pred_index = list(model.classes_).index(prediction)
                        
                        if isinstance(shap_values, list):
                            class_shap_values = shap_values[pred_index]
                        else:
                            if len(shap_values.shape) == 3:
                                class_shap_values = shap_values[:, :, pred_index]
                            else:
                                class_shap_values = shap_values
                                
                        fig, ax = plt.subplots(figsize=(8, 4))
                        
                        shap.summary_plot(class_shap_values, input_df, plot_type="bar", feature_names=display_features, show=False)
                        
                        # Customize matplotlib plot for light theme
                        fig.patch.set_facecolor('#ffffff')
                        ax.set_facecolor('#ffffff')
                        ax.tick_params(colors='#111827')
                        ax.xaxis.label.set_color('#111827')
                        ax.yaxis.label.set_color('#111827')
                        
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Could not generate SHAP explanation: {str(e)}")

if __name__ == "__main__":
    main()
