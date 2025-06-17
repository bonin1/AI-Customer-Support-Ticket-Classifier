"""
Main Streamlit application for customer support ticket classification.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from predict import TicketClassifier, load_classifier_from_directory, ModelEvaluator
    from data_preprocessing import TextPreprocessor
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .high-confidence {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .medium-confidence {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .low-confidence {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def load_model_list():
    """Load list of available models."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models')
    if not os.path.exists(models_dir):
        return []
    
    models = []
    files = os.listdir(models_dir)
    
    # Find model files and extract names
    model_files = [f for f in files if f.endswith(('.h5', '.pkl', '.keras'))]
    
    for model_file in model_files:
        model_name = model_file.rsplit('.', 1)[0]
        # Check if preprocessor exists
        preprocessor_file = f"{model_name}_preprocessor.pkl"
        if preprocessor_file in files:
            models.append(model_name)
    
    return sorted(models, reverse=True)  # Most recent first

@st.cache_resource
def load_classifier(model_name: str):
    """Load classifier with caching."""
    try:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models')
        classifier = load_classifier_from_directory(models_dir, model_name)
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence level."""
    if confidence >= 0.8:
        return "high-confidence"
    elif confidence >= 0.5:
        return "medium-confidence"
    else:
        return "low-confidence"

def format_confidence(confidence: float) -> str:
    """Format confidence as percentage."""
    return f"{confidence:.1%}"

def create_confidence_gauge(confidence: float, title: str = "Confidence"):
    """Create a confidence gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def display_single_prediction(result: Dict[str, Any]):
    """Display single prediction result."""
    confidence_class = get_confidence_class(result['confidence'])
    
    st.markdown(f"""
    <div class="prediction-result {confidence_class}">
        <h3>🎯 Predicted Category: {result['predicted_category']}</h3>
        <p><strong>Confidence:</strong> {format_confidence(result['confidence'])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show confidence gauge
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = create_confidence_gauge(result['confidence'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'probabilities' in result:
            st.subheader("Category Probabilities")
            prob_df = pd.DataFrame([
                {'Category': cat, 'Probability': prob}
                for cat, prob in result['probabilities'].items()
            ]).sort_values('Probability', ascending=False)
            
            fig = px.bar(prob_df, x='Probability', y='Category', orientation='h',
                        title="Probability Distribution")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎫 AI Customer Support Ticket Classifier</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    available_models = load_model_list()
    
    if not available_models:
        st.error("No trained models found. Please train a model first using the training script.")
        st.info("Run: `python src/train.py --data_path data/sample/sample_tickets.csv --model_type lstm`")
        return
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        help="Choose a trained model for classification"
    )
    
    # Load classifier
    classifier = load_classifier(selected_model)
    if classifier is None:
        st.error("Failed to load the selected model.")
        return
    
    # Model info
    with st.sidebar.expander("Model Information"):
        model_info = classifier.get_model_info()
        st.json(model_info)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Single Prediction",
        "📋 Batch Processing", 
        "📊 Analytics Dashboard",
        "🔧 Model Management"
    ])
    
    with tab1:
        st.header("Single Ticket Classification")
        
        # Text input
        ticket_text = st.text_area(
            "Enter ticket message:",
            height=150,
            placeholder="Example: My credit card was charged twice for the same order. Can you please refund the duplicate charge?"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            predict_button = st.button("🎯 Classify Ticket", type="primary")
        
        if predict_button and ticket_text.strip():
            with st.spinner("Classifying ticket..."):
                result = classifier.predict_single(ticket_text, return_probabilities=True)
                
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    display_single_prediction(result)
                    
                    # Show processed text
                    with st.expander("Processed Text"):
                        st.code(result.get('processed_text', ''))
        
        elif predict_button:
            st.warning("Please enter a ticket message.")
    
    with tab2:
        st.header("Batch Ticket Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with tickets",
            type=['csv'],
            help="CSV should contain a 'customer_message' column"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} tickets")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Column selection
                text_columns = df.columns.tolist()
                selected_column = st.selectbox(
                    "Select text column:",
                    text_columns,
                    index=text_columns.index('customer_message') if 'customer_message' in text_columns else 0
                )
                
                # Process button
                if st.button("🚀 Process All Tickets", type="primary"):
                    with st.spinner("Processing tickets..."):
                        progress_bar = st.progress(0)
                        
                        # Process in batches for better performance
                        batch_size = 10
                        results_df = df.copy()
                        
                        for i in range(0, len(df), batch_size):
                            batch_end = min(i + batch_size, len(df))
                            batch_texts = df[selected_column].iloc[i:batch_end].tolist()
                            
                            batch_results = classifier.predict_batch(batch_texts, return_probabilities=True)
                            
                            # Update dataframe
                            for j, result in enumerate(batch_results):
                                idx = i + j
                                results_df.loc[idx, 'predicted_category'] = result['predicted_category']
                                results_df.loc[idx, 'confidence'] = result['confidence']
                            
                            progress_bar.progress((batch_end) / len(df))
                        
                        st.success("Processing completed!")
                        
                        # Show results
                        st.subheader("Classification Results")
                        st.dataframe(results_df)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"classified_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Quick analytics
                        st.subheader("Quick Analytics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_confidence = results_df['confidence'].mean()
                            st.metric("Average Confidence", f"{avg_confidence:.1%}")
                        
                        with col2:
                            high_conf_count = len(results_df[results_df['confidence'] >= 0.8])
                            st.metric("High Confidence", f"{high_conf_count} ({high_conf_count/len(results_df):.1%})")
                        
                        with col3:
                            most_common = results_df['predicted_category'].value_counts().iloc[0]
                            most_common_cat = results_df['predicted_category'].value_counts().index[0]
                            st.metric("Most Common Category", f"{most_common_cat} ({most_common})")
                        
                        # Category distribution chart
                        category_counts = results_df['predicted_category'].value_counts()
                        fig = px.pie(values=category_counts.values, names=category_counts.index,
                                   title="Category Distribution")
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        # Sample data for demonstration
        st.info("Upload and process tickets in the Batch Processing tab to see analytics here.")
        
        # If we have session state with processed data, show analytics
        if 'processed_df' in st.session_state:
            df = st.session_state.processed_df
            
            # Key metrics
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tickets", len(df))
            with col2:
                avg_conf = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            with col3:
                high_conf = len(df[df['confidence'] >= 0.8])
                st.metric("High Confidence", f"{high_conf}")
            with col4:
                categories = df['predicted_category'].nunique()
                st.metric("Categories", categories)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Category distribution
                cat_counts = df['predicted_category'].value_counts()
                fig = px.bar(x=cat_counts.values, y=cat_counts.index, orientation='h',
                           title="Tickets by Category")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig = px.histogram(df, x='confidence', nbins=20,
                                 title="Confidence Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Show sample analytics with dummy data
            st.subheader("Sample Analytics")
            
            # Create sample data
            sample_data = {
                'Category': ['Billing', 'Technical Issue', 'Feature Request', 'Account Management', 'General Inquiry'],
                'Count': [45, 32, 18, 25, 15],
                'Avg Confidence': [0.85, 0.78, 0.92, 0.73, 0.68]
            }
            sample_df = pd.DataFrame(sample_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(sample_df, x='Count', y='Category', orientation='h',
                           title="Sample Category Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(sample_df, x='Category', y='Avg Confidence',
                           title="Average Confidence by Category")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Model Management")
        
        # Model comparison
        st.subheader("Available Models")
        
        models_data = []
        for model_name in available_models:
            try:
                temp_classifier = load_classifier(model_name)
                info = temp_classifier.get_model_info()
                models_data.append({
                    'Model Name': model_name,
                    'Type': info.get('model_type', 'Unknown'),
                    'Accuracy': info.get('test_accuracy', 'N/A'),
                    'Classes': info.get('num_classes', 'N/A'),
                    'Training Samples': info.get('training_samples', 'N/A')
                })
            except Exception:
                models_data.append({
                    'Model Name': model_name,
                    'Type': 'Unknown',
                    'Accuracy': 'N/A',
                    'Classes': 'N/A',
                    'Training Samples': 'N/A'
                })
        
        if models_data:
            models_df = pd.DataFrame(models_data)
            st.dataframe(models_df, use_container_width=True)
        
        # Human-in-the-loop correction
        st.subheader("Human-in-the-Loop Corrections")
        st.info("Feature coming soon: Manual correction interface for improving model performance.")
        
        # Model retraining
        st.subheader("Model Retraining")
        if st.button("🔄 Retrain Model"):
            st.info("Feature coming soon: Automated model retraining with corrected data.")

if __name__ == "__main__":
    main()
