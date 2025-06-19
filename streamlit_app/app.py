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
import time
import logging
from typing import Dict, Any, List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from predict import TicketClassifier, load_classifier_from_directory, ModelEvaluator
    from data_preprocessing import TextPreprocessor
    from ensemble_predictor import EnsemblePredictor
    from active_learning import ActiveLearningOracle
    from model_explainer import ModelExplainer
    from hyperparameter_tuning import HyperparameterTuner
    from drift_detector import DataDriftDetector
    from model_versioning import ModelVersionManager
    from feature_engineering import AdvancedFeatureEngineer
    from online_learner import OnlineLearner
    from audit_system import MLAuditSystem
    from response_generator import AIResponseGenerator
    from streaming_processor import StreamingProcessor, StreamingTicket, WebSocketStreaming
    from multimodal_classifier import MultiModalClassifier, MultiModalInput
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
    
    # Initialize session state early
    if 'streaming_results' not in st.session_state:
        st.session_state.streaming_results = []
    if 'multimodal_results' not in st.session_state:
        st.session_state.multimodal_results = []
    
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
    )    # Load classifier
    classifier = load_classifier(selected_model)
    if classifier is None:
        st.error("Failed to load the selected model.")
        return
    
    # Store classifier in session state for use in tab functions
    # Clear existing processors if model changed
    if st.session_state.get('selected_model') != selected_model:
        # Clear processors that depend on the classifier
        keys_to_clear = ['streaming_processor', 'multimodal_classifier', 'response_generator']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    st.session_state.classifier = classifier
    st.session_state.selected_model = selected_model
      # Model info
    with st.sidebar.expander("Model Information"):
        model_info = classifier.get_model_info()
        st.json(model_info)
    
    # Navigation menu in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("Navigation")
    
    selected_tab = st.sidebar.selectbox(
        "Select Feature",
        [
            "🔍 Single Prediction",
            "📋 Batch Processing", 
            "📊 Analytics Dashboard",
            "🔧 Model Management",
            "🕵️ Data Drift Monitor",
            "🔄 Online Learning",
            "🏗️ Feature Engineering",
            "📋 Audit & Compliance",
            "🤖 AI Response Generator",
            "📡 Real-time Streaming",
            "🎭 Multi-modal Processing"
        ],
        index=0
    )
      # Render selected tab content
    if selected_tab == "🔍 Single Prediction":
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
        
        elif predict_button:            st.warning("Please enter a ticket message.")
    
    elif selected_tab == "📋 Batch Processing":
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
    
    elif selected_tab == "📊 Analytics Dashboard":
        st.header("📊 Comprehensive Analytics Dashboard")
        
        # Load all available ticket data
        @st.cache_data
        def load_all_ticket_data():
            """Load all ticket data from available sources."""
            all_data = []
            data_sources = []
            
            # Define data file paths
            data_files = [
                ("Sample Tickets", "data/sample/sample_tickets.csv"),
                ("Training Data", "data/sample/train_tickets.csv"),
                ("Test Data", "data/sample/test_tickets.csv")
            ]
            
            for source_name, file_path in data_files:
                try:
                    full_path = os.path.join(os.path.dirname(__file__), '..', file_path)
                    if os.path.exists(full_path):
                        df = pd.read_csv(full_path)
                        df['data_source'] = source_name
                        df['file_location'] = file_path
                        all_data.append(df)
                        data_sources.append({
                            'source': source_name,
                            'path': file_path,
                            'count': len(df),
                            'status': 'Loaded'
                        })
                    else:
                        data_sources.append({
                            'source': source_name,
                            'path': file_path,
                            'count': 0,
                            'status': 'Not Found'
                        })
                except Exception as e:
                    data_sources.append({
                        'source': source_name,
                        'path': file_path,
                        'count': 0,
                        'status': f'Error: {str(e)}'
                    })
            
            # Check for processed data in session state
            if 'processed_df' in st.session_state:
                processed_df = st.session_state.processed_df.copy()
                processed_df['data_source'] = 'Processed (Session)'
                processed_df['file_location'] = 'session_state'
                all_data.append(processed_df)
                data_sources.append({
                    'source': 'Processed (Session)',
                    'path': 'session_state',
                    'count': len(processed_df),
                    'status': 'Loaded'
                })
            
            combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
            return combined_df, data_sources
        
        # Load data
        with st.spinner("Loading ticket data..."):
            combined_df, data_sources = load_all_ticket_data()
        
        # Data Source Overview
        st.subheader("📁 Data Sources Overview")
        
        if data_sources:
            sources_df = pd.DataFrame(data_sources)
            
            # Display data sources table
            st.dataframe(sources_df, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sources = len(data_sources)
                st.metric("Total Sources", total_sources)
            
            with col2:
                loaded_sources = len([s for s in data_sources if s['status'] == 'Loaded'])
                st.metric("Loaded Sources", loaded_sources)
            
            with col3:
                total_tickets = sum([s['count'] for s in data_sources if isinstance(s['count'], int)])
                st.metric("Total Tickets", total_tickets)
            
            with col4:
                if combined_df.empty:
                    st.metric("Data Status", "No Data")
                else:
                    st.metric("Data Status", "✅ Ready")
        
        if not combined_df.empty:
            st.divider()
            
            # Overall Statistics
            st.subheader("📈 Overall Statistics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Tickets", len(combined_df))
            
            with col2:
                unique_categories = combined_df['category'].nunique()
                st.metric("Unique Categories", unique_categories)
            
            with col3:
                if 'priority' in combined_df.columns:
                    high_priority = len(combined_df[combined_df['priority'] == 'High'])
                    st.metric("High Priority", high_priority)
                else:
                    st.metric("High Priority", "N/A")
            
            with col4:
                if 'channel' in combined_df.columns:
                    unique_channels = combined_df['channel'].nunique()
                    st.metric("Channels", unique_channels)
                else:
                    st.metric("Channels", "N/A")
            
            with col5:
                date_range = "N/A"
                if 'timestamp' in combined_df.columns:
                    try:
                        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
                        date_range = f"{(combined_df['timestamp'].max() - combined_df['timestamp'].min()).days} days"
                    except:
                        pass
                st.metric("Date Range", date_range)
            
            # Visualizations
            st.subheader("📊 Data Visualizations")
            
            # Category Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Category Distribution**")
                cat_counts = combined_df['category'].value_counts()
                fig = px.pie(values=cat_counts.values, names=cat_counts.index, 
                           title="Tickets by Category")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Data Source Distribution**")
                source_counts = combined_df['data_source'].value_counts()
                fig = px.bar(x=source_counts.values, y=source_counts.index, 
                           orientation='h', title="Tickets by Data Source")
                st.plotly_chart(fig, use_container_width=True)
            
            # Priority and Channel Analysis (if available)
            if 'priority' in combined_df.columns and 'channel' in combined_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Priority Distribution**")
                    priority_counts = combined_df['priority'].value_counts()
                    fig = px.bar(x=priority_counts.index, y=priority_counts.values,
                               title="Tickets by Priority")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Channel Distribution**")
                    channel_counts = combined_df['channel'].value_counts()
                    fig = px.bar(x=channel_counts.index, y=channel_counts.values,
                               title="Tickets by Channel")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Time Analysis (if timestamp available)
            if 'timestamp' in combined_df.columns:
                try:
                    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
                    combined_df['date'] = combined_df['timestamp'].dt.date
                    combined_df['hour'] = combined_df['timestamp'].dt.hour
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Tickets Over Time**")
                        daily_counts = combined_df['date'].value_counts().sort_index()
                        fig = px.line(x=daily_counts.index, y=daily_counts.values,
                                    title="Daily Ticket Volume")
                        fig.update_xaxis(title="Date")
                        fig.update_yaxis(title="Number of Tickets")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Tickets by Hour of Day**")
                        hourly_counts = combined_df['hour'].value_counts().sort_index()
                        fig = px.bar(x=hourly_counts.index, y=hourly_counts.values,
                                   title="Ticket Volume by Hour")
                        fig.update_xaxis(title="Hour of Day")
                        fig.update_yaxis(title="Number of Tickets")
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Could not parse timestamp data for time analysis.")
            
            # Category-Priority Heatmap (if both available)
            if 'priority' in combined_df.columns and 'category' in combined_df.columns:
                st.write("**Category vs Priority Heatmap**")
                heatmap_data = pd.crosstab(combined_df['category'], combined_df['priority'])
                fig = px.imshow(heatmap_data.values, 
                              labels=dict(x="Priority", y="Category", color="Count"),
                              x=heatmap_data.columns, 
                              y=heatmap_data.index,
                              title="Category vs Priority Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Raw Data Display
            st.subheader("🗂️ All Tickets - Raw Data")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Category filter
                categories = ['All'] + sorted(combined_df['category'].unique().tolist())
                selected_category = st.selectbox("Filter by Category", categories)
            
            with col2:
                # Data source filter
                sources = ['All'] + sorted(combined_df['data_source'].unique().tolist())
                selected_source = st.selectbox("Filter by Data Source", sources)
            
            with col3:
                # Priority filter (if available)
                if 'priority' in combined_df.columns:
                    priorities = ['All'] + sorted(combined_df['priority'].unique().tolist())
                    selected_priority = st.selectbox("Filter by Priority", priorities)
                else:
                    selected_priority = 'All'
            
            # Apply filters
            filtered_df = combined_df.copy()
            
            if selected_category != 'All':
                filtered_df = filtered_df[filtered_df['category'] == selected_category]
            
            if selected_source != 'All':
                filtered_df = filtered_df[filtered_df['data_source'] == selected_source]
            
            if selected_priority != 'All' and 'priority' in combined_df.columns:
                filtered_df = filtered_df[filtered_df['priority'] == selected_priority]
            
            # Display filtered results
            st.write(f"**Showing {len(filtered_df)} of {len(combined_df)} tickets**")
            
            # Column selection for display
            available_columns = filtered_df.columns.tolist()
            default_columns = [col for col in ['ticket_id', 'customer_message', 'category', 'priority', 'channel', 'timestamp', 'data_source'] if col in available_columns]
            
            selected_columns = st.multiselect(
                "Select columns to display",
                available_columns,
                default=default_columns
            )
            
            if selected_columns:
                display_df = filtered_df[selected_columns].copy()
                
                # Format for better display
                if 'customer_message' in display_df.columns:
                    display_df['customer_message'] = display_df['customer_message'].apply(
                        lambda x: x[:100] + "..." if len(str(x)) > 100 else x
                    )
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Download filtered data
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=csv,
                    file_name=f"filtered_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Detailed ticket view
            st.divider()
            st.subheader("🔍 Detailed Ticket View")
            
            if len(filtered_df) > 0:
                ticket_ids = filtered_df['ticket_id'].tolist() if 'ticket_id' in filtered_df.columns else list(range(len(filtered_df)))
                selected_ticket_id = st.selectbox("Select ticket to view details", ticket_ids)
                
                if 'ticket_id' in filtered_df.columns:
                    selected_ticket = filtered_df[filtered_df['ticket_id'] == selected_ticket_id].iloc[0]
                else:
                    selected_ticket = filtered_df.iloc[selected_ticket_id]
                
                # Display ticket details
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Full Message:**")
                    st.text_area("", value=selected_ticket.get('customer_message', 'N/A'), height=150, disabled=True)
                
                with col2:
                    st.write("**Ticket Information:**")
                    for col in selected_ticket.index:
                        if col != 'customer_message':
                            st.write(f"**{col.replace('_', ' ').title()}:** {selected_ticket[col]}")
        
        else:
            st.warning("No ticket data found. Please ensure data files are available in the data directory or process some tickets in the Batch Processing tab.")
            
            # Show expected data structure
            st.subheader("Expected Data Structure")
            st.write("The system looks for CSV files with the following structure:")
            
            expected_structure = pd.DataFrame({
                'Column': ['ticket_id', 'customer_message', 'category', 'timestamp', 'priority', 'channel'],
                'Description': [
                    'Unique identifier for the ticket',
                    'The customer\'s message/complaint',
                    'Category of the ticket',
                    'When the ticket was created',
                    'Priority level (High, Medium, Low)',
                    'Communication channel (Email, Chat, Phone)'
                ],                'Required': ['Yes', 'Yes', 'Yes', 'No', 'No', 'No']
            })
            
            st.dataframe(expected_structure, use_container_width=True)
    
    elif selected_tab == "🔧 Model Management":
        model_management_tab()

    elif selected_tab == "🕵️ Data Drift Monitor":
        data_drift_monitor_tab()

    elif selected_tab == "🔄 Online Learning":
        online_learning_tab()

    elif selected_tab == "🏗️ Feature Engineering":        
        feature_engineering_tab()
        
    elif selected_tab == "📋 Audit & Compliance":
        audit_compliance_tab()

    elif selected_tab == "🤖 AI Response Generator":
        ai_response_generator_tab()
    
    elif selected_tab == "📡 Real-time Streaming":
        streaming_processing_tab()
    
    elif selected_tab == "🎭 Multi-modal Processing":
        multimodal_processing_tab()

# Advanced feature functions

    @st.cache_resource
    def load_ensemble_predictor():
        """Load ensemble predictor"""
        try:
            ensemble = EnsemblePredictor()
            if ensemble.load_all_models():
                return ensemble
        except Exception as e:
            st.error(f"Error loading ensemble predictor: {str(e)}")
        return None

    def ensemble_prediction_tab(ensemble):
        """Enhanced ensemble prediction tab"""
        st.header("🤖 Ensemble Predictions")
        st.write("Get predictions from multiple models with confidence analysis")
        
        if ensemble is None:
            st.warning("Ensemble predictor not available. Multiple models needed.")
            return
        
        # Text input
        ticket_text = st.text_area(
            "Enter your support ticket:",
            height=150,
            placeholder="Describe your issue here...",
            key="ensemble_text"
        )
        
        if st.button("🔮 Get Ensemble Prediction", type="primary"):
            if ticket_text.strip():
                with st.spinner("Running ensemble prediction..."):
                    try:
                        result = ensemble.predict_with_confidence(ticket_text)
                        
                        # Main prediction
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.success(f"**Ensemble Prediction:** {result['ensemble_prediction']}")
                            st.metric("Ensemble Confidence", f"{result['ensemble_confidence']:.2%}")
                            st.metric("Model Agreement", f"{result['agreement_score']:.2%}")
                        
                        with col2:
                            confidence_color = "🟢" if result['ensemble_confidence'] > 0.8 else "🟡" if result['ensemble_confidence'] > 0.6 else "🔴"
                            st.markdown(f"## {confidence_color}")
                            
                            if result['high_confidence']:
                                st.success("High Confidence")
                            else:
                                st.warning("Low Confidence")
                        
                        # Individual model predictions
                        st.subheader("Individual Model Predictions")
                        
                        individual_data = []
                        for model_name, prediction in result['individual_predictions'].items():
                            confidence = result['individual_confidences'].get(model_name, 0)
                            individual_data.append({
                                'Model': model_name,
                                'Prediction': prediction,
                                'Confidence': f"{confidence:.2%}"
                            })
                        
                        if individual_data:
                            df = pd.DataFrame(individual_data)
                            st.dataframe(df, use_container_width=True)
                        
                        # Detailed probabilities
                        with st.expander("📊 Detailed Probability Analysis"):
                            st.write("**Ensemble Probabilities:**")
                            ensemble_probs = result.get('ensemble_probabilities', {})
                            if ensemble_probs:
                                prob_df = pd.DataFrame([
                                    {'Category': k, 'Probability': f"{v:.2%}"} 
                                    for k, v in ensemble_probs.items()
                                ])
                                st.dataframe(prob_df)
                            
                            st.write("**Individual Model Probabilities:**")
                            for model_name, probs in result.get('individual_probabilities', {}).items():
                                st.write(f"*{model_name}:*")
                                if isinstance(probs, list) and len(probs) > 0:
                                    # Assume order matches class names
                                    class_names = ['Billing', 'Technical Issue', 'Feature Request', 
                                                 'Account Management', 'Product Information', 
                                                 'Refund & Return', 'General Inquiry']
                                    prob_dict = dict(zip(class_names[:len(probs)], probs))
                                    st.json(prob_dict)
                    
                    except Exception as e:
                        st.error(f"Error in ensemble prediction: {str(e)}")
            else:
                st.warning("Please enter a ticket description.")

    def explainability_tab(explainer, ensemble):
        """Model explainability and interpretability tab"""
        st.header("🔍 Model Explainability")
        st.write("Understand how the model makes decisions")
        
        # Text input for explanation
        text_to_explain = st.text_area(
            "Enter text to explain:",
            height=100,
            placeholder="Enter a support ticket to see how the model interprets it...",
            key="explain_text"
        )
        
        # Explanation type selection
        explain_type = st.selectbox(
            "Explanation Type:",
            ["LIME Local Explanation", "Feature Importance", "Counterfactual Examples"]
        )
        
        if st.button("🔍 Generate Explanation"):
            if text_to_explain.strip():
                with st.spinner("Generating explanation..."):
                    try:
                        if explain_type == "LIME Local Explanation":
                            if ensemble:
                                explanation = explainer.explain_prediction_lime(text_to_explain, ensemble)
                                
                                if 'error' not in explanation:
                                    st.subheader("LIME Explanation")
                                    st.write("**Local feature importance for this prediction:**")
                                    
                                    # Display explanation
                                    local_exp = explanation.get('local_explanation', [])
                                    if local_exp:
                                        exp_df = pd.DataFrame(local_exp, columns=['Feature', 'Importance'])
                                        exp_df['Importance'] = exp_df['Importance'].astype(float)
                                        
                                        # Color code positive/negative importance
                                        fig = px.bar(
                                            exp_df, 
                                            x='Importance', 
                                            y='Feature',
                                            orientation='h',
                                            color='Importance',
                                            color_continuous_scale='RdBu_r',
                                            title='Feature Importance for Prediction'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display prediction probabilities
                                    pred_probs = explanation.get('prediction', [])
                                    if pred_probs:
                                        st.write("**Prediction Probabilities:**")
                                        class_names = explainer._get_class_names()
                                        prob_dict = dict(zip(class_names[:len(pred_probs)], pred_probs))
                                        st.json(prob_dict)
                                else:
                                    st.error(f"Error generating LIME explanation: {explanation['error']}")
                            else:
                                st.warning("Ensemble model required for LIME explanations")
                        
                        elif explain_type == "Feature Importance":
                            st.subheader("Global Feature Importance")
                            st.info("Feature importance analysis requires training data. This would show the most important words/phrases for each category.")
                            

                            # Placeholder for feature importance
                            sample_features = {
                                'Billing': ['payment', 'charge', 'invoice', 'refund', 'subscription'],
                                'Technical Issue': ['error', 'bug', 'crash', 'broken', 'not working'],
                                'Feature Request': ['feature', 'suggest', 'enhancement', 'add', 'improvement']
                            }
                            
                            for category, features in sample_features.items():
                                st.write(f"**{category}:** {', '.join(features)}")
                        
                        elif explain_type == "Counterfactual Examples":
                            if ensemble:
                                st.subheader("Counterfactual Explanations")
                                st.write("*What minimal changes would flip the prediction?*")
                                
                                counterfactuals = explainer.generate_counterfactual_examples(
                                    text_to_explain, ensemble
                                )
                                
                                if 'error' not in counterfactuals:
                                    original_class = counterfactuals.get('original_prediction', 'Unknown')
                                    target_class = counterfactuals.get('target_class', 'Unknown')
                                    
                                    st.write(f"**Original prediction:** {original_class}")
                                    st.write(f"**Target class:** {target_class}")
                                    
                                    cf_examples = counterfactuals.get('counterfactuals', [])
                                    if cf_examples:
                                        for i, cf in enumerate(cf_examples[:3]):  # Show top 3
                                            with st.expander(f"Counterfactual {i+1}: {cf['change']}"):
                                                st.write(f"**Original:** {cf['original_text']}")
                                                st.write(f"**Modified:** {cf['modified_text']}")
                                                st.write(f"**New Prediction:** {cf['new_class']}")
                                                st.write(f"**Confidence Change:** {cf['confidence_change']:+.2%}")
                                    else:
                                        st.info("No counterfactual examples found")
                                else:
                                    st.error(f"Error generating counterfactuals: {counterfactuals['error']}")
                            else:
                                st.warning("Ensemble model required for counterfactual explanations")
                    
                    except Exception as e:
                        st.error(f"Error generating explanation: {str(e)}")
            else:
                st.warning("Please enter text to explain.")
        
        # Model bias analysis section
        st.subheader("🎯 Bias Analysis")
        if st.button("Run Bias Analysis"):
            with st.spinner("Analyzing potential bias..."):
                try:
                    # This would require actual training data
                    st.info("Bias analysis requires access to training data with sensitive attributes.")
                    st.write("**Potential bias indicators to monitor:**")
                    st.write("- Prediction differences based on language style")
                    st.write("- Performance variations across user demographics")  
                    st.write("- Systematic misclassification patterns")
                    
                    # Placeholder bias results
                    bias_results = {
                        'urgency_bias': 0.05,
                        'gender_bias': 0.02,
                        'age_bias': 0.03
                    }
                    
                    for bias_type, score in bias_results.items():
                        color = "🟢" if score < 0.05 else "🟡" if score < 0.1 else "🔴"
                        st.write(f"{color} **{bias_type.replace('_', ' ').title()}:** {score:.3f}")
                
                except Exception as e:
                    st.error(f"Error in bias analysis: {str(e)}")

    def active_learning_tab(oracle, ensemble):
        """Active learning and human-in-the-loop tab"""
        st.header("⚡ Active Learning & Human Feedback")
        st.write("Improve model performance through targeted human feedback")
        
        # Sample uncertain predictions section
        st.subheader("🎯 Uncertain Predictions")
        st.write("Review predictions that the model is uncertain about")
        
        # Generate sample uncertain predictions
        sample_texts = [
            "The app is slow sometimes",
            "Need help with something",
            "Having trouble with the new update"
        ]
        
        if st.button("🔍 Find Uncertain Predictions"):
            if ensemble:
                with st.spinner("Analyzing predictions for uncertainty..."):
                    try:
                        predictions = []
                        for text in sample_texts:
                            result = ensemble.predict_with_confidence(text)
                            predictions.append(result)
                        
                        uncertain_samples = oracle.identify_uncertain_samples(predictions, sample_texts)
                        
                        if uncertain_samples:
                            st.success(f"Found {len(uncertain_samples)} uncertain predictions")
                            
                            for i, sample in enumerate(uncertain_samples[:3]):  # Show top 3
                                with st.expander(f"Uncertain Sample {i+1} (Score: {sample['uncertainty_score']:.3f})"):
                                    st.write(f"**Text:** {sample['text']}")
                                    st.write(f"**Current Prediction:** {sample['prediction']}")
                                    st.write(f"**Confidence:** {sample['confidence']:.2%}")
                                    
                                    # Human correction interface
                                    st.write("**Provide Correction:**")
                                    correct_category = st.selectbox(
                                        "Correct category:",
                                        ['Billing', 'Technical Issue', 'Feature Request', 
                                         'Account Management', 'Product Information', 
                                         'Refund & Return', 'General Inquiry'],
                                        key=f"correct_{i}"
                                    )
                                    
                                    if st.button(f"Submit Correction {i+1}", key=f"submit_{i}"):
                                        # Record feedback
                                        oracle.record_human_feedback(
                                            sample_index=sample['index'],
                                            original_text=sample['text'],
                                            predicted_label=sample['prediction'],
                                            correct_label=correct_category,
                                            confidence=sample['confidence']
                                        )
                                        st.success("Feedback recorded!")
                        else:
                            st.info("No highly uncertain predictions found")
                            
                    except Exception as e:
                        st.error(f"Error finding uncertain predictions: {str(e)}")
            else:
                st.warning("Ensemble model required for uncertainty analysis")
        
        # Feedback summary
        st.subheader("📊 Feedback Summary")
        
        # Mock feedback data
        feedback_summary = {
            'total_feedback': len(oracle.feedback_data),
            'accuracy_from_feedback': oracle._calculate_feedback_accuracy() if oracle.feedback_data else 0.0,
            'common_errors': oracle.analyze_error_patterns() if oracle.feedback_data else {'error_patterns': {}}
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Feedback", feedback_summary['total_feedback'])
        
        with col2:
            st.metric("Feedback Accuracy", f"{feedback_summary['accuracy_from_feedback']:.1%}")
        
        with col3:
            error_patterns = feedback_summary['common_errors']['error_patterns']
            st.metric("Error Patterns", len(error_patterns))
        
        # Retraining recommendations
        if oracle.feedback_data:
            st.subheader("🔄 Retraining Recommendations")
            
            retraining_strategy = oracle.suggest_retraining_strategy()
            
            if retraining_strategy['should_retrain']:
                st.warning("Model retraining recommended!")
                
                for strategy in retraining_strategy['strategies']:
                    st.write(f"- **{strategy['type'].replace('_', ' ').title()}**: {strategy['target']}")
                
                if st.button("🚀 Start Retraining"):
                    st.info("Retraining feature would be implemented here")
            else:
                st.success("No retraining needed yet")
                st.write(retraining_strategy['reason'])
        
        # Export feedback data
        if oracle.feedback_data:
            st.subheader("📤 Export Feedback Data")
            
            if st.button("Export Training Data"):
                # Would export corrected samples for retraining
                st.info("Training data export feature would save corrected samples to file")

    def single_prediction_tab(classifier):
        """Original single prediction tab"""
        st.header("Single Ticket Classification")
        
        # Text input
        ticket_text = st.text_area(
            "Enter your support ticket:",
            height=150,
            placeholder="Describe your issue here..."
        )
        
        # Classification options
        col1, col2 = st.columns(2)
        
        with col1:
            include_confidence = st.checkbox("Show confidence scores", value=True)
        
        with col2:
            include_explanation = st.checkbox("Show explanation", value=False)
        
        if st.button("🔍 Classify Ticket", type="primary"):
            if ticket_text.strip():
                with st.spinner("Analyzing ticket..."):
                    try:
                        # Make prediction
                        prediction = classifier.predict(ticket_text)
                        confidence = prediction['confidence']
                        category = prediction['predicted_class']
                        
                        # Display result
                        st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.success(f"**Predicted Category:** {category}")
                            
                            if include_confidence:
                                # Confidence meter
                                confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                                st.markdown(f"""
                                <div style="background-color: {confidence_color}; 
                                           width: {confidence*100}%; 
                                           height: 20px; 
                                           border-radius: 10px; 
                                           margin: 10px 0;">
                                </div>
                                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            # Category icon/color
                            category_icons = {
                                "Billing": "💳",
                                "Technical Issue": "🔧",
                                "Feature Request": "💡",
                                "Account Management": "👤",
                                "Product Information": "📋",
                                "Refund & Return": "↩️",
                                "General Inquiry": "❓"
                            }
                            icon = category_icons.get(category, "🎫")
                            st.markdown(f"<div style='font-size: 4rem; text-align: center;'>{icon}</div>", 
                                      unsafe_allow_html=True)
                        
                        # Additional information
                        if include_explanation:
                            with st.expander("📊 Detailed Analysis"):
                                st.write("**Model Information:**")
                                info = classifier.get_model_info()
                                st.json(info)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
            else:
                st.warning("Please enter a ticket description.")

    def batch_processing_tab(classifier):
        """Original batch processing tab with enhancements"""
        st.header("Batch Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with tickets",
            type=['csv'],
            help="CSV should have a 'text' column with ticket descriptions"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} tickets")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Select text column
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                text_column = st.selectbox("Select text column:", text_columns)
                
                if st.button("🚀 Process All Tickets"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    for i, text in enumerate(df[text_column]):
                        try:
                            prediction = classifier.predict(str(text))
                            results.append({
                                'Original Text': text,
                                'Predicted Category': prediction['predicted_class'],
                                'Confidence': prediction['confidence']
                            })
                        except Exception as e:
                            results.append({
                                'Original Text': text,
                                'Predicted Category': 'Error',
                                'Confidence': 0.0
                            })
                        
                        # Update progress
                        progress = (i + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing ticket {i+1}/{len(df)}")
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.subheader("Classification Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_confidence = results_df['Confidence'].mean()
                        st.metric("Average Confidence", f"{avg_confidence:.2%}")
                    
                    with col2:
                        high_conf_count = len(results_df[results_df['Confidence'] > 0.8])
                        st.metric("High Confidence", f"{high_conf_count}/{len(results_df)}")
                    
                    with col3:
                        unique_categories = results_df['Predicted Category'].nunique()
                        st.metric("Unique Categories", unique_categories)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    def analytics_dashboard_tab():
        """Enhanced analytics dashboard"""
        st.header("Analytics Dashboard")
        
        # Load sample data for visualization
        try:
            from utils.dashboard_utils import create_sample_analytics_data, create_category_distribution_chart, create_confidence_distribution_chart
            
            # Generate sample data
            analytics_data = create_sample_analytics_data()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tickets", analytics_data['total_tickets'])
            
            with col2:
                st.metric("Avg Confidence", f"{analytics_data['avg_confidence']:.2%}")
            
            with col3:
                st.metric("Most Common", analytics_data['most_common_category'])
            
            with col4:
                st.metric("High Confidence", f"{analytics_data['high_confidence_rate']:.1%}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Category Distribution")
                fig1 = create_category_distribution_chart(analytics_data['category_counts'])
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.subheader("Confidence Distribution")
                fig2 = create_confidence_distribution_chart(analytics_data['confidence_scores'])
                st.plotly_chart(fig2, use_container_width=True)
            
            # Time series (if available)
            st.subheader("Ticket Volume Over Time")
            time_data = analytics_data.get('time_series', [])
            if time_data:
                time_df = pd.DataFrame(time_data)
                fig3 = px.line(time_df, x='date', y='count', title='Daily Ticket Volume')
                st.plotly_chart(fig3, use_container_width=True)
            else:                st.info("No time series data available")
                
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")

def get_available_models():
    """Get list of available trained models"""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models')
    if not os.path.exists(models_dir):
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5') or f.endswith('.pkl')]
    model_names = []
    
    for file in model_files:
        if '_' in file:
            model_name = file.split('_')[0]
            if model_name not in model_names:
                model_names.append(model_name)    
    return model_names

def model_management_tab():
    """Model management and comparison tab."""
    st.header("🔧 Model Management")
    
    available_models = load_model_list()
    
    if not available_models:
        st.warning("No trained models found.")
        return
    
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
        
        # Hyperparameter tuning section
        st.subheader("🎛️ Hyperparameter Tuning")
        
        model_type = st.selectbox(
            "Select model type for tuning:",
            ['lstm', 'cnn', 'random_forest', 'svm', 'logistic_regression']
        )
        
        n_trials = st.slider("Number of trials:", 10, 100, 20)
        
        if st.button("🎯 Start Hyperparameter Tuning"):
            with st.spinner("Running hyperparameter optimization..."):
                try:
                    tuner = HyperparameterTuner()
                    st.info(f"Starting {n_trials} trials for {model_type} model...")
                    st.info("This is a demonstration. In practice, this would run actual optimization.")
                    
                    # Mock results
                    best_params = {
                        'lstm': {'embedding_dim': 128, 'lstm_units': 64, 'dropout_rate': 0.3},
                        'random_forest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5}
                    }.get(model_type, {})
                    
                    st.success("Hyperparameter tuning completed!")
                    st.write("**Best Parameters:**")
                    st.json(best_params)
                    
                except Exception as e:
                    st.error(f"Error in hyperparameter tuning: {str(e)}")
        
        # Model deployment section
        st.subheader("🚀 Model Deployment")
        
        if st.button("Deploy Best Model"):
            st.info("Model deployment feature would update the active model")
        
        # Model retraining
        st.subheader("🔄 Model Retraining")
        retrain_options = st.multiselect(
            "Retraining options:",
            ["Include human feedback", "Use latest data", "Hyperparameter optimization"]
        )
        
        if st.button("Start Retraining"):
            if retrain_options:
                st.info(f"Starting retraining with options: {', '.join(retrain_options)}")
            else:
                st.warning("Please select retraining options")

def data_drift_monitor_tab():
    """Data drift monitoring tab."""
    st.header("🕵️ Data Drift Monitor")
    
    st.markdown("""
    Monitor your model's input data for distribution changes that might affect performance.
    """)
    
    # Initialize drift detector
    if 'drift_detector' not in st.session_state:
        st.session_state.drift_detector = DataDriftDetector()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Reference Data")
        uploaded_file = st.file_uploader(
            "Upload reference/baseline data (CSV)",
            type=['csv'],
            help="Upload historical data to establish baseline distribution"
        )
        
        if uploaded_file is not None:
            reference_data = pd.read_csv(uploaded_file)
            st.write("Reference data preview:")
            st.dataframe(reference_data.head())
            
            if st.button("Fit Reference Data"):
                with st.spinner("Fitting reference data..."):
                    try:
                        if 'text' in reference_data.columns:
                            texts = reference_data['text'].tolist()
                            labels = reference_data.get('category', None)
                            if labels is not None:
                                labels = labels.tolist()
                            
                            st.session_state.drift_detector.fit_reference_data(texts, labels)
                            st.success("Reference data fitted successfully!")
                        else:
                            st.error("Reference data must have a 'text' column")
                    except Exception as e:
                        st.error(f"Error fitting reference data: {e}")
    
    with col2:
        st.subheader("Drift Detection Settings")
        drift_threshold = st.slider("Drift Threshold", 0.01, 0.2, 0.05, 0.01)
        window_size = st.number_input("Window Size", 100, 5000, 1000)
        
        if st.button("Update Settings"):
            st.session_state.drift_detector.drift_threshold = drift_threshold
            st.session_state.drift_detector.window_size = window_size
            st.success("Settings updated!")
    
    # Drift detection section
    st.subheader("Check for Data Drift")
    
    # Option 1: Upload new data
    new_data_file = st.file_uploader(
        "Upload new data to check for drift (CSV)",
        type=['csv'],
        key="drift_check_file"
    )
    
    # Option 2: Enter text manually
    manual_texts = st.text_area(
        "Or enter texts manually (one per line):",
        height=100,
        placeholder="Enter ticket texts, one per line..."
    )
    
    if st.button("Detect Drift"):
        try:
            if new_data_file is not None:
                new_data = pd.read_csv(new_data_file)
                if 'text' in new_data.columns:
                    texts = new_data['text'].tolist()
                    labels = new_data.get('category', None)
                    if labels is not None:
                        labels = labels.tolist()
                else:
                    st.error("New data must have a 'text' column")
                    return
            elif manual_texts.strip():
                texts = [text.strip() for text in manual_texts.split('\n') if text.strip()]
                labels = None
            else:
                st.warning("Please upload data or enter texts manually")
                return
            
            with st.spinner("Detecting drift..."):
                drift_report = st.session_state.drift_detector.detect_drift(texts, labels)
                
                if 'error' in drift_report:
                    st.error(f"Error: {drift_report['error']}")
                else:
                    # Display drift results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        drift_status = "🚨 DETECTED" if drift_report['drift_detected'] else "✅ NONE"
                        st.metric("Drift Status", drift_status)
                    
                    with col2:
                        st.metric("Drift Score", f"{drift_report['drift_score']:.3f}")
                    
                    with col3:
                        st.metric("Sample Size", drift_report['sample_size'])
                    
                    # Detailed results
                    st.subheader("Drift Detection Methods")
                    methods_df = []
                    for method, result in drift_report['drift_methods'].items():
                        if 'error' not in result:
                            methods_df.append({
                                'Method': method.upper(),
                                'Drift Detected': '✅' if result.get('drift_detected', False) else '❌',
                                'Details': str(result)[:100] + '...' if len(str(result)) > 100 else str(result)
                            })
                    
                    if methods_df:
                        st.dataframe(pd.DataFrame(methods_df))
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    for rec in drift_report.get('recommendations', []):
                        st.write(f"• {rec}")
        
        except Exception as e:
            st.error(f"Error detecting drift: {e}")
      # Drift history
    st.subheader("Drift Detection History")
    
    if 'drift_detector' in st.session_state:
        drift_summary = st.session_state.drift_detector.get_drift_summary()
        st.json(drift_summary)
    else:
        st.info("Drift detector not initialized yet.")

def online_learning_tab():
    """Online learning tab."""
    st.header("🔄 Online Learning System")
    
    st.markdown("""
    Continuously improve your model with user feedback and real-time learning.
    """)
    
    # Initialize online learner
    if 'online_learner' not in st.session_state:
        st.session_state.online_learner = OnlineLearner()
    
    # Learning statistics
    col1, col2, col3, col4 = st.columns(4)
    
    stats = st.session_state.online_learner.get_learning_stats()
    
    with col1:
        st.metric("Total Predictions", stats['total_predictions'])
    
    with col2:
        st.metric("Feedback Received", stats['feedback_received'])
    
    with col3:
        st.metric("Model Updates", stats['model_updates'])
    
    with col4:
        accuracy = stats['current_accuracy']
        st.metric("Current Accuracy", f"{accuracy:.1%}" if accuracy > 0 else "N/A")
    
    # Feedback collection
    st.subheader("Provide Feedback")
    
    with st.form("feedback_form"):
        text_input = st.text_area("Ticket Text", height=100)
        predicted_category = st.selectbox("Predicted Category", 
                                        ["Billing", "Technical", "General", "Complaint", "Compliment"])
        true_category = st.selectbox("Correct Category", 
                                   ["Billing", "Technical", "General", "Complaint", "Compliment"])
        confidence = st.slider("Model Confidence", 0.0, 1.0, 0.8)
        user_id = st.text_input("User ID (optional)")
        
        if st.form_submit_button("Submit Feedback"):
            if text_input.strip():
                success = st.session_state.online_learner.add_feedback(
                    text_input,
                    predicted_category,
                    true_category,
                    confidence,
                    user_id if user_id else None
                )
                
                if success:
                    st.success("Feedback submitted successfully!")
                else:
                    st.error("Error submitting feedback")
            else:
                st.warning("Please enter ticket text")
    
    # Learning settings
    st.subheader("Learning Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        batch_size = st.number_input("Batch Size", 5, 100, 10)
    
    with col2:
        update_frequency = st.number_input("Update Frequency", 10, 500, 50)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7)
    
    if st.button("Update Learning Settings"):
        st.session_state.online_learner.learning_rate = learning_rate
        st.session_state.online_learner.batch_size = batch_size
        st.session_state.online_learner.update_frequency = update_frequency
        st.session_state.online_learner.confidence_threshold = confidence_threshold
        st.success("Settings updated!")
    
    # Feedback summary
    st.subheader("Feedback Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        days = st.selectbox("Time Period", [7, 14, 30], index=0)
        
        if st.button("Generate Summary"):
            try:
                summary = st.session_state.online_learner.get_feedback_summary(days)
                st.json(summary)
            except Exception as e:
                st.error(f"Error generating summary: {e}")
    
    with col2:
        if st.button("Export Feedback Data"):
            try:
                filepath = f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.session_state.online_learner.export_feedback_data(filepath, days=30)
                st.success(f"Data exported to {filepath}")
            except Exception as e:
                st.error(f"Error exporting data: {e}")

def feature_engineering_tab():
    """Advanced feature engineering tab."""
    st.header("🏗️ Advanced Feature Engineering")
    
    st.markdown("""
    Create sophisticated features from your text data using advanced NLP techniques.
    """)
    
    # Initialize feature engineer
    if 'feature_engineer' not in st.session_state:
        st.session_state.feature_engineer = AdvancedFeatureEngineer()
      # Feature engineering settings
    st.subheader("Feature Engineering Configuration")
    
    st.info("""
    **Parameter Tips:**
    - **Min Document Frequency**: For small samples (< 5 texts), use 1 to avoid filtering all terms
    - **Max Document Frequency**: 0.95 filters very common words (appearing in >95% of texts)
    - **N-grams**: (1,3) captures single words, bigrams, and trigrams
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        max_features = st.number_input("Max TF-IDF Features", 1000, 50000, 10000)
        min_df = st.number_input("Min Document Frequency", 1, 10, 1)
        max_df = st.slider("Max Document Frequency", 0.5, 1.0, 0.95)
    
    with col2:
        ngram_min = st.number_input("N-gram Min", 1, 3, 1)
        ngram_max = st.number_input("N-gram Max", 1, 5, 3)
        num_topics = st.number_input("Number of Topics (LDA)", 5, 50, 20)
    
    # Sample data for demonstration
    st.subheader("Try Feature Engineering")
    
    sample_texts = st.text_area(
        "Enter sample texts (one per line):",
        value="I can't log into my account\nMy bill is incorrect\nGreat service, thank you!",
        height=100
    )
    
    if st.button("Extract Features"):
        if sample_texts.strip():
            texts = [text.strip() for text in sample_texts.split('\n') if text.strip()]
            
            with st.spinner("Extracting features..."):
                try:
                    # Configure feature engineer
                    st.session_state.feature_engineer.max_features = max_features
                    st.session_state.feature_engineer.min_df = min_df
                    st.session_state.feature_engineer.max_df = max_df
                    st.session_state.feature_engineer.ngram_range = (ngram_min, ngram_max)
                    st.session_state.feature_engineer.num_topics = num_topics
                    
                    # Adjust parameters for small samples to avoid pruning all terms
                    num_texts = len(texts)
                    if num_texts <= 5 and min_df > 1:
                        st.warning(f"Small sample size ({num_texts} texts). Adjusting min_df to 1 to avoid over-pruning.")
                        st.session_state.feature_engineer.min_df = 1
                    
                    # Fit and transform
                    st.session_state.feature_engineer.fit(texts)
                    features = st.session_state.feature_engineer.transform(texts)
                    
                    # Display results
                    st.success(f"Extracted {features.shape[1]} features from {features.shape[0]} texts")
                    
                    # Feature categories
                    st.subheader("Feature Categories")
                    feature_categories = st.session_state.feature_engineer.get_top_features_by_category(n_features=10)
                    
                    for category, feature_list in feature_categories.items():
                        if feature_list:
                            with st.expander(f"{category.title()} Features ({len(feature_list)})"):
                                st.write(feature_list[:10])  # Show top 10
                    
                    # Feature matrix preview
                    st.subheader("Feature Matrix Preview")
                    feature_df = pd.DataFrame(
                        features[:, :10],  # Show first 10 features
                        columns=st.session_state.feature_engineer.feature_names[:10],
                        index=[f"Text {i+1}" for i in range(len(texts))]
                    )
                    st.dataframe(feature_df)
                    
                except Exception as e:
                    st.error(f"Error extracting features: {e}")
        else:
            st.warning("Please enter some sample texts")
    
    # Feature analysis
    st.subheader("Feature Analysis Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Analyze Feature Importance"):
            st.info("Feature importance requires a trained model. Train a model first.")
    
    with col2:
        if st.button("Export Feature Pipeline"):
            try:
                filepath = f"feature_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                st.session_state.feature_engineer.save_pipeline(filepath)
                st.success(f"Pipeline saved to {filepath}")
            except Exception as e:
                st.error(f"Error saving pipeline: {e}")

def audit_compliance_tab():
    """Audit and compliance tab."""
    st.header("📋 Audit & Compliance Dashboard")
    
    st.markdown("""
    Monitor model usage, ensure compliance, and maintain audit trails.
    """)
    
    # Initialize audit system
    if 'audit_system' not in st.session_state:
        st.session_state.audit_system = MLAuditSystem()
    
    # Compliance status
    st.subheader("Compliance Status")
    
    compliance_status = st.session_state.audit_system.get_compliance_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Rules", len([r for r in compliance_status['compliance_rules'].values() if r['enabled']]))
    
    with col2:
        st.metric("Unresolved Violations", compliance_status['unresolved_violations'])
    
    with col3:
        st.metric("Data Retention (Days)", compliance_status['data_retention_days'])
    
    # Audit report
    st.subheader("Audit Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        end_date = st.date_input("End Date", datetime.now())
    
    with col2:
        event_types = st.multiselect(
            "Event Types", 
            ["prediction", "training", "access", "model_update"],
            default=["prediction"]
        )
    
    if st.button("Generate Audit Report"):
        try:
            report = st.session_state.audit_system.get_audit_report(
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.max.time()),
                event_types=event_types if event_types else None
            )
            
            # Display summary
            st.subheader("Report Summary")
            summary_cols = st.columns(4)
            
            with summary_cols[0]:
                st.metric("Total Events", report['summary']['total_events'])
            
            with summary_cols[1]:
                st.metric("Unique Users", report['summary']['unique_users'])
            
            with summary_cols[2]:
                st.metric("Models Accessed", report['summary']['models_accessed'])
            
            with summary_cols[3]:
                st.metric("Violations", report['summary']['compliance_violations'])
            
            # Event statistics
            if report['event_statistics']:
                st.subheader("Event Statistics")
                event_df = pd.DataFrame(list(report['event_statistics'].items()), 
                                      columns=['Event Type', 'Count'])
                fig = px.bar(event_df, x='Event Type', y='Count', title="Events by Type")
                st.plotly_chart(fig, use_container_width=True)
            
            # Top users
            if report['top_users']:
                st.subheader("Top Users")
                users_df = pd.DataFrame(list(report['top_users'].items()), 
                                      columns=['User ID', 'Activity Count'])
                st.dataframe(users_df)
            
            # Full report
            with st.expander("Full Report Details"):
                st.json(report)
        
        except Exception as e:
            st.error(f"Error generating audit report: {e}")
    
    # GDPR section
    st.subheader("GDPR Compliance")
    
    if compliance_status['gdpr_enabled']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Create GDPR Request**")
            request_type = st.selectbox("Request Type", 
                                      ["access", "deletion", "portability"])
            user_id = st.text_input("User ID")
            email = st.text_input("Email")
            
            if st.button("Create GDPR Request"):
                if user_id or email:
                    try:
                        request_id = st.session_state.audit_system.create_gdpr_request(
                            request_type, user_id, email
                        )
                        st.success(f"GDPR request created: {request_id}")
                    except Exception as e:
                        st.error(f"Error creating request: {e}")
                else:
                    st.warning("Please provide either User ID or Email")
        
        with col2:
            st.write("**Process GDPR Request**")
            request_id = st.text_input("Request ID to Process")
            
            if st.button("Process Request"):
                if request_id:
                    try:
                        result = st.session_state.audit_system.process_gdpr_request(request_id)
                        st.json(result)
                    except Exception as e:
                        st.error(f"Error processing request: {e}")
                else:
                    st.warning("Please provide Request ID")
    else:
        st.info("GDPR features are not enabled")
    
    # Data cleanup
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clean Expired Data"):
            try:
                deleted_count = st.session_state.audit_system.cleanup_expired_data()
                st.success(f"Deleted {deleted_count} expired records")
            except Exception as e:
                st.error(f"Error cleaning data: {e}")
    
    with col2:
        if st.button("Export Audit Data"):
            try:
                filepath = f"audit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.session_state.audit_system.export_audit_data(filepath)
                st.success(f"Audit data exported to {filepath}")
            except Exception as e:
                st.error(f"Error exporting data: {e}")

def ai_response_generator_tab():
    """AI Response Generator tab."""
    st.header("🤖 AI Response Generator")
    
    st.markdown("""
    Generate intelligent responses to customer support tickets using open-source LLMs from Hugging Face.
    """)
    
    # Initialize response generator
    if 'response_generator' not in st.session_state:
        with st.spinner("Initializing AI Response Generator..."):
            try:
                st.session_state.response_generator = AIResponseGenerator()
                st.success("AI Response Generator initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing AI Response Generator: {str(e)}")
                st.session_state.response_generator = None
    
    response_generator = st.session_state.response_generator
    
    if response_generator is None:
        st.warning("AI Response Generator is not available. Please check the installation.")
        return
    
    # Configuration Section
    st.subheader("Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model selection
        model_options = [
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium", 
            "google/flan-t5-small",
            "google/flan-t5-base",
            "facebook/blenderbot-400M-distill",
            "gpt2",
            "gpt2-medium"
        ]
        
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=0,
            help="Choose the Hugging Face model for response generation"
        )
        
        # Generation method
        generation_method = st.selectbox(
            "Generation Method",
            ["hybrid", "template", "ai"],
            index=0,
            help="Hybrid uses both templates and AI, Template uses predefined templates, AI uses pure model generation"
        )
    
    with col2:
        # Generation parameters
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in generation (lower = more deterministic)"
        )
        
        max_length = st.slider(
            "Max Response Length",
            min_value=50,
            max_value=300,
            value=150,
            step=10,
            help="Maximum length of generated responses"
        )
    
    # Switch model if needed
    if selected_model != response_generator.model_name:
        if st.button("Switch Model"):
            with st.spinner(f"Switching to {selected_model}..."):
                try:
                    response_generator.switch_model(selected_model)
                    st.success(f"Switched to {selected_model}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error switching model: {str(e)}")
    
    st.divider()
    
    # Response Generation Section
    st.subheader("Generate Response")
    
    # Input fields
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticket_text = st.text_area(
            "Customer Ticket",
            height=150,
            placeholder="Enter the customer support ticket text here...",
            help="Paste or type the customer's support ticket"
        )
    
    with col2:
        customer_name = st.text_input(
            "Customer Name",
            value="Customer",
            help="Customer's name for personalization"
        )
        
        predicted_category = st.selectbox(
            "Ticket Category",
            ["Technical Support", "Billing", "Account", "Product", "General Inquiry", "Complaint"],
            help="Select or predict the ticket category"
        )
        
        urgency = st.selectbox(
            "Urgency Level",
            ["low", "medium", "high"],
            index=1,
            help="Select the urgency level"
        )
        
        confidence = st.slider(
            "Prediction Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Confidence score for the prediction"
        )
    
    # Generate button
    if st.button("Generate Response", type="primary", use_container_width=True):
        if not ticket_text.strip():
            st.warning("Please enter a ticket text.")
        else:
            with st.spinner("Generating response..."):
                try:
                    # Update generator parameters
                    response_generator.temperature = temperature
                    response_generator.max_length = max_length
                    
                    # Generate response
                    response = response_generator.generate_response(
                        ticket_text=ticket_text,
                        predicted_category=predicted_category,
                        confidence=confidence,
                        urgency=urgency,
                        customer_name=customer_name,
                        generation_method=generation_method
                    )
                    
                    # Display results
                    st.success("Response generated successfully!")
                    
                    # Response display
                    st.subheader("Generated Response")
                    st.write(response.response_text)
                    
                    # Response metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Confidence Score", f"{response.confidence_score:.2f}")
                    
                    with col2:
                        st.metric("Method", response.generation_method.title())
                    
                    with col3:
                        st.metric("Template Used", response.template_used)
                    
                    with col4:
                        escalation_color = "red" if response.escalation_recommended else "green"
                        st.metric("Escalation", "Yes" if response.escalation_recommended else "No")
                    
                    # Suggested actions
                    if response.suggested_actions:
                        st.subheader("Suggested Actions")
                        for i, action in enumerate(response.suggested_actions, 1):
                            st.write(f"{i}. {action}")
                    
                    # Metadata
                    with st.expander("Response Metadata"):
                        st.json(response.metadata)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.exception(e)
    
    st.divider()
    
    # Batch Processing Section
    st.subheader("Batch Response Generation")
    
    st.markdown("Upload a CSV file with ticket data for batch processing.")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type="csv",
        help="CSV should contain columns: ticket_text, category, urgency, customer_name"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            required_columns = ['ticket_text']
            optional_columns = ['category', 'urgency', 'customer_name']
            
            if all(col in df.columns for col in required_columns):
                if st.button("Generate Batch Responses"):
                    with st.spinner("Processing batch requests..."):
                        try:
                            # Prepare batch data
                            batch_data = []
                            for _, row in df.iterrows():
                                batch_data.append({
                                    'ticket_text': row['ticket_text'],
                                    'category': row.get('category', 'General Inquiry'),
                                    'urgency': row.get('urgency', 'medium'),
                                    'customer_name': row.get('customer_name', 'Customer')
                                })
                            
                            # Generate responses
                            responses = response_generator.batch_generate_responses(
                                batch_data, generation_method=generation_method
                            )
                            
                            # Create results DataFrame
                            results_df = pd.DataFrame([
                                {
                                    'Original_Ticket': data['ticket_text'],
                                    'Generated_Response': resp.response_text,
                                    'Confidence': resp.confidence_score,
                                    'Method': resp.generation_method,
                                    'Template': resp.template_used,
                                    'Escalation_Recommended': resp.escalation_recommended
                                }
                                for data, resp in zip(batch_data, responses)
                            ])
                            
                            st.success(f"Generated {len(responses)} responses!")
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name=f"batch_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error processing batch: {str(e)}")
            else:
                st.error(f"CSV must contain columns: {required_columns}")
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
    
    st.divider()
    
    # Statistics Section
    st.subheader("Response Statistics")
    
    try:
        stats = response_generator.get_response_statistics()
        
        if stats['total_responses'] > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Responses", stats['total_responses'])
            
            with col2:
                st.metric("Average Confidence", f"{stats['average_confidence']:.2f}")
            
            with col3:
                st.metric("Escalation Rate", f"{stats['escalation_rate']:.1f}%")
            
            # Method distribution
            if stats['method_distribution']:
                st.subheader("Generation Method Distribution")
                method_df = pd.DataFrame(list(stats['method_distribution'].items()), 
                                       columns=['Method', 'Count'])
                fig = px.pie(method_df, values='Count', names='Method', 
                           title="Response Generation Methods")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No responses generated yet. Generate some responses to see statistics.")
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")

def streaming_processing_tab():
    """Real-time streaming processing tab."""
    st.header("📡 Real-time Streaming Processing")
    
    st.markdown("""
    Process customer support tickets in real-time using streaming data sources.
    """)
      # Initialize streaming processor
    if 'streaming_processor' not in st.session_state:
        try:
            classifier = st.session_state.get('classifier')
            if not classifier:
                st.error("No classifier available. Please load a model first.")
                st.info("Go to the 'Single Prediction' tab to load a model.")
                return
                
            response_generator = st.session_state.get('response_generator')
            st.session_state.streaming_processor = StreamingProcessor(
                classifier=classifier,
                response_generator=response_generator,
                enable_monitoring=True,
                enable_drift_detection=True
            )
            st.success("Streaming processor initialized!")
        except Exception as e:
            st.error(f"Error initializing streaming processor: {str(e)}")
            return
    
    streaming_processor = st.session_state.streaming_processor
    
    # Configuration Section
    st.subheader("Streaming Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Processing Settings**")
        batch_size = st.slider("Batch Size", min_value=1, max_value=50, value=10)
        batch_timeout = st.slider("Batch Timeout (seconds)", min_value=1.0, max_value=30.0, value=5.0)
        
        # Update processor settings
        streaming_processor.batch_size = batch_size
        streaming_processor.batch_timeout = batch_timeout
    
    with col2:
        st.write("**Streaming Sources**")
        enable_websocket = st.checkbox("WebSocket Server", value=False)
        websocket_port = st.number_input("WebSocket Port", min_value=1000, max_value=9999, value=8765)
        
        enable_simulation = st.checkbox("Ticket Simulation", value=True)
        simulation_interval = st.slider("Simulation Interval (seconds)", min_value=1, max_value=30, value=5)
      # Control Section
    st.subheader("Streaming Control")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Streaming", type="primary"):
            try:
                # Initialize streaming results before starting
                if 'streaming_results' not in st.session_state:
                    st.session_state.streaming_results = []
                
                streaming_processor.start_streaming()
                st.success("Streaming started!")
                st.info("Add tickets manually below to see results.")
                
            except Exception as e:
                st.error(f"Error starting streaming: {str(e)}")
    
    with col2:
        if st.button("⏹️ Stop Streaming"):
            try:
                streaming_processor.stop_streaming()
                st.success("Streaming stopped!")
            except Exception as e:
                st.error(f"Error stopping streaming: {str(e)}")
    
    with col3:
        if st.button("🔄 Reset"):
            if 'streaming_results' in st.session_state:
                st.session_state.streaming_results = []
            st.success("Results cleared!")
    
    # Status Section
    st.subheader("Streaming Status")
    
    stats = streaming_processor.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", "Running" if stats['is_running'] else "Stopped")
    
    with col2:
        st.metric("Total Processed", stats['total_processed'])
    
    with col3:
        st.metric("Queue Size", stats['queue_size'])
    
    with col4:
        if stats.get('throughput_per_second'):
            st.metric("Throughput/sec", f"{stats['throughput_per_second']:.1f}")
        else:
            st.metric("Avg Processing Time", f"{stats['avg_processing_time']:.1f}ms")
    
    # Manual Ticket Injection
    st.subheader("Manual Ticket Input")
    
    with st.form("manual_ticket_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticket_text = st.text_area("Ticket Message", height=100)
        
        with col2:            
            customer_id = st.text_input("Customer ID", value="test_customer")
            priority = st.selectbox("Priority", ["low", "medium", "high"])
            channel = st.selectbox("Channel", ["manual", "email", "chat", "phone"])
        
        if st.form_submit_button("Add Ticket to Stream"):
            if ticket_text.strip():
                try:
                    # Initialize streaming results if not exists
                    if 'streaming_results' not in st.session_state:
                        st.session_state.streaming_results = []
                    
                    # Process ticket directly (synchronous for manual input)
                    ticket = StreamingTicket(
                        ticket_id=f"manual_{int(time.time())}",
                        customer_message=ticket_text,
                        timestamp=datetime.now(),
                        channel=channel,
                        priority=priority,
                        customer_id=customer_id
                    )
                    
                    # Get the classifier and process the ticket
                    classifier = st.session_state.get('classifier')
                    if classifier:
                        with st.spinner("Processing ticket..."):
                            # Process the ticket directly
                            import time as time_module
                            start_time = time_module.time()
                            
                            prediction = classifier.predict_single(ticket.customer_message, return_probabilities=True)
                            processing_time = (time_module.time() - start_time) * 1000
                            
                            # Create result
                            from streaming_processor import StreamingResult
                            result = StreamingResult(
                                ticket_id=ticket.ticket_id,
                                predicted_category=prediction.get('predicted_category', 'Unknown'),
                                confidence=prediction.get('confidence', 0.0),
                                processing_time_ms=processing_time,
                                escalation_required=prediction.get('confidence', 0.0) < 0.7,
                                drift_detected=False,
                                timestamp=datetime.now()
                            )
                            
                            # Store result in session state
                            st.session_state.streaming_results.append(result)
                            if len(st.session_state.streaming_results) > 100:
                                st.session_state.streaming_results = st.session_state.streaming_results[-100:]
                    
                    st.success("Ticket processed and added to results!")
                    st.rerun()  # Refresh to show the new result
                    
                except Exception as e:
                    st.error(f"Error processing ticket: {str(e)}")
                    st.exception(e)
            else:
                st.warning("Please enter a ticket message.")
    
    # Results Display
    st.subheader("Recent Results")
    
    if 'streaming_results' in st.session_state and st.session_state.streaming_results:
        results = st.session_state.streaming_results[-20:]  # Show last 20 results
        
        # Create DataFrame for display
        results_data = []
        for result in reversed(results):  # Most recent first
            results_data.append({
                'Ticket ID': result.ticket_id,
                'Category': result.predicted_category,
                'Confidence': f"{result.confidence:.2f}",
                'Processing Time (ms)': f"{result.processing_time_ms:.1f}",
                'Escalation': "Yes" if result.escalation_required else "No",
                'Drift Detected': "Yes" if result.drift_detected else "No",
                'Timestamp': result.timestamp.strftime('%H:%M:%S') if result.timestamp else 'N/A'
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        if st.button("📥 Download Results"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"streaming_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No streaming results yet. Start streaming and add some tickets to see results here.")

def multimodal_processing_tab():
    """Multi-modal processing tab."""
    st.header("🎭 Multi-modal Processing")
    
    st.markdown("""
    Process customer support tickets with multiple data types: text, images, audio, and documents.
    """)
      # Initialize multi-modal classifier
    if 'multimodal_classifier' not in st.session_state:
        try:
            classifier = st.session_state.get('classifier')
            if not classifier:
                st.error("No text classifier available. Please load a model first.")
                st.info("Go to the 'Single Prediction' tab to load a model.")
                return
                
            st.session_state.multimodal_classifier = MultiModalClassifier(
                text_classifier=classifier,
                enable_ocr=True,
                enable_image_classification=True,
                enable_audio_processing=True,
                enable_document_parsing=True
            )
            st.success("Multi-modal classifier initialized!")
        except Exception as e:
            st.error(f"Error initializing multi-modal classifier: {str(e)}")
            return
    
    multimodal_classifier = st.session_state.multimodal_classifier
    
    # System Capabilities
    st.subheader("System Capabilities")
    
    capabilities = multimodal_classifier.get_capabilities()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Text Processing", "✅" if capabilities['text_processing'] else "❌")
        st.metric("Image Processing", "✅" if capabilities['image_processing'] else "❌")
    
    with col2:
        st.metric("OCR", "✅" if capabilities['ocr'] else "❌")
        st.metric("Image Classification", "✅" if capabilities['image_classification'] else "❌")
    
    with col3:
        st.metric("Audio Processing", "✅" if capabilities['audio_processing'] else "❌")
        st.metric("Document Parsing", "✅" if capabilities['document_parsing'] else "❌")
    
    with col4:
        st.metric("Vision Models", "✅" if capabilities['vision_models'] else "❌")
        st.metric("Streaming Support", "✅" if capabilities['streaming_support'] else "❌")
    
    st.divider()
    
    # Multi-modal Input Section
    st.subheader("Multi-modal Input")
    
    # Text input
    st.write("**Text Message**")
    text_input = st.text_area("Customer message", height=100, 
                             placeholder="Enter the customer's message here...")
    
    # Image input
    st.write("**Images**")
    uploaded_images = st.file_uploader(
        "Upload images (screenshots, photos, receipts, etc.)",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        accept_multiple_files=True,
        help="Upload up to 5 images for analysis"
    )
    
    # Audio input
    st.write("**Audio**")
    uploaded_audio = st.file_uploader(
        "Upload audio file (voicemail, recorded call, etc.)",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Audio will be processed for transcription and analysis"
    )
    
    # Document input
    st.write("**Documents**")
    uploaded_documents = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT files)",
        type=['pdf', 'docx', 'txt', 'log', 'csv'],
        accept_multiple_files=True,
        help="Upload up to 10 documents for text extraction"
    )
    
    # Processing button
    if st.button("🔍 Process Multi-modal Input", type="primary"):
        if not any([text_input.strip(), uploaded_images, uploaded_audio, uploaded_documents]):
            st.warning("Please provide at least one type of input.")
        else:
            with st.spinner("Processing multi-modal input..."):
                try:
                    # Prepare input data
                    input_data = MultiModalInput()
                    
                    # Text
                    if text_input.strip():
                        input_data.text = text_input
                    
                    # Images
                    if uploaded_images:
                        input_data.images = []
                        for img_file in uploaded_images[:5]:  # Limit to 5 images
                            # Save temporarily and add path
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{img_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(img_file.getvalue())
                                input_data.images.append(tmp_file.name)
                    
                    # Audio
                    if uploaded_audio:
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_audio.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_audio.getvalue())
                            input_data.audio = tmp_file.name
                    
                    # Documents
                    if uploaded_documents:
                        input_data.attachments = []
                        for doc_file in uploaded_documents[:10]:  # Limit to 10 documents
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(doc_file.getvalue())
                                input_data.attachments.append(tmp_file.name)
                    
                    # Process
                    result = multimodal_classifier.classify_multimodal(input_data)
                    
                    # Display results
                    st.success("Multi-modal processing completed!")
                    
                    # Main prediction
                    st.subheader("Classification Result")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Category", result.predicted_category)
                    
                    with col2:
                        st.metric("Confidence", f"{result.confidence:.2f}")
                    
                    with col3:
                        st.metric("Processing Time", f"{result.processing_details['processing_time_seconds']:.2f}s")
                    
                    # Modal contributions
                    st.subheader("Modal Contributions")
                    
                    if result.modal_contributions:
                        contrib_df = pd.DataFrame(
                            list(result.modal_contributions.items()),
                            columns=['Modality', 'Contribution']
                        )
                        
                        fig = px.bar(contrib_df, x='Modality', y='Contribution',
                                   title="Contribution of Each Modality")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Extracted features
                    st.subheader("Extracted Features")
                    
                    with st.expander("Text Features"):
                        if result.extracted_features.text_features:
                            st.json(result.extracted_features.text_features)
                        else:
                            st.info("No text features extracted")
                    
                    with st.expander("Image Features"):
                        if result.extracted_features.image_features:
                            features = result.extracted_features.image_features
                            
                            if features.get('extracted_text'):
                                st.write("**Extracted Text from Images:**")
                                st.write(features['extracted_text'])
                            
                            if features.get('descriptions'):
                                st.write("**Image Descriptions:**")
                                for i, desc in enumerate(features['descriptions'], 1):
                                    st.write(f"{i}. {desc}")
                            
                            if features.get('classifications'):
                                st.write("**Image Classifications:**")
                                for i, classification in enumerate(features['classifications'], 1):
                                    st.write(f"{i}. {classification['label']} (confidence: {classification['confidence']:.2f})")
                        else:
                            st.info("No image features extracted")
                    with st.expander("Audio Features"):
                        if result.extracted_features.audio_features:
                            st.json(result.extracted_features.audio_features)
                        else:
                            st.info("No audio features extracted")
                    
                    with st.expander("Document Features"):
                        if result.extracted_features.document_features:
                            features = result.extracted_features.document_features
                            
                            # Show document processing capabilities
                            if 'capabilities' in features:
                                st.write("**Document Processing Capabilities:**")
                                caps = features['capabilities']
                                cap_col1, cap_col2, cap_col3 = st.columns(3)
                                with cap_col1:
                                    st.write(f"PDF: {'✅' if caps.get('pdf_support') else '❌'}")
                                with cap_col2:
                                    st.write(f"DOCX: {'✅' if caps.get('docx_support') else '❌'}")
                                with cap_col3:
                                    st.write(f"OCR: {'✅' if caps.get('ocr_support') else '❌'}")
                            
                            # Show processing statistics
                            if 'processed_count' in features:
                                st.write(f"**Processing Statistics:**")
                                stat_col1, stat_col2, stat_col3 = st.columns(3)
                                with stat_col1:
                                    st.metric("Documents Uploaded", features.get('document_count', 0))
                                with stat_col2:
                                    st.metric("Successfully Processed", features.get('processed_count', 0))
                                with stat_col3:
                                    st.metric("Processing Confidence", f"{features.get('confidence', 0):.2f}")
                            
                            # Show warnings if any
                            if 'warnings' in features and features['warnings']:
                                st.warning("**Processing Warnings:**")
                                for warning in features['warnings']:
                                    st.write(f"⚠️ {warning}")
                            
                            # Show extracted text
                            if features.get('extracted_text'):
                                st.write("**Extracted Text from Documents:**")
                                text = features['extracted_text']
                                display_text = text[:1000] + "..." if len(text) > 1000 else text
                                st.text_area("", display_text, height=200, disabled=True)
                                
                                if len(text) > 1000:
                                    st.info(f"Showing first 1000 characters of {len(text)} total characters")
                            
                            # Show document types
                            if features.get('document_types'):
                                st.write("**Document Types Detected:**")
                                for i, doc_type in enumerate(features['document_types'], 1):
                                    st.write(f"{i}. {doc_type}")
                        else:
                            st.info("No document features extracted")
                            
                            # Show helpful information about document processing
                            st.write("**Document Processing Requirements:**")
                            st.write("- PDF files: Requires `PyMuPDF` (pip install PyMuPDF)")
                            st.write("- DOCX files: Requires `python-docx` (pip install python-docx)")
                            st.write("- Image files: Requires OCR support (tesseract or easyocr)")
                            st.write("- Text files: Supported natively (.txt, .log, .csv, .json, .xml)")
                    
                    # Combined text
                    with st.expander("Combined Text for Classification"):
                        st.text_area("", result.extracted_features.combined_text, height=150, disabled=True)
                    
                    # Processing details
                    with st.expander("Processing Details"):
                        st.json(result.processing_details)
                    
                    # Cleanup temporary files
                    try:
                        if input_data.images:
                            for img_path in input_data.images:
                                if os.path.exists(img_path):
                                    os.unlink(img_path)
                        
                        if input_data.audio and os.path.exists(input_data.audio):
                            os.unlink(input_data.audio)
                        
                        if input_data.attachments:
                            for doc_path in input_data.attachments:
                                if os.path.exists(doc_path):
                                    os.unlink(doc_path)
                    except:
                        pass  # Ignore cleanup errors
                    
                except Exception as e:
                    st.error(f"Error processing multi-modal input: {str(e)}")
                    st.exception(e)
    
    st.divider()
    
    # Statistics
    st.subheader("Processing Statistics")
    
    stats = multimodal_classifier.get_processing_stats()
    
    if stats['total_processed'] > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Processed", stats['total_processed'])
            
            usage_stats = stats['modality_usage']
            
            st.write("**Modality Usage:**")
            for modality, count in usage_stats.items():
                if count > 0:
                    percentage = (count / stats['total_processed']) * 100
                    st.write(f"- {modality.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        with col2:
            # Create pie chart of modality usage
            usage_data = [(k.replace('_', ' ').title(), v) for k, v in stats['modality_usage'].items() if v > 0]
            
            if usage_data:
                usage_df = pd.DataFrame(usage_data, columns=['Type', 'Count'])
                fig = px.pie(usage_df, values='Count', names='Type', 
                           title="Modality Usage Distribution")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No multi-modal processing performed yet.")

if __name__ == "__main__":
    main()
