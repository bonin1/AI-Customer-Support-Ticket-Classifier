"""
Utility functions for the Streamlit application.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import streamlit as st

def create_category_distribution_chart(df: pd.DataFrame, column: str = 'predicted_category') -> go.Figure:
    """
    Create a category distribution chart.
    
    Args:
        df: DataFrame with predictions
        column: Column containing categories
        
    Returns:
        Plotly figure
    """
    category_counts = df[column].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Category Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_confidence_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a confidence distribution histogram.
    
    Args:
        df: DataFrame with confidence scores
        
    Returns:
        Plotly figure
    """
    fig = px.histogram(
        df,
        x='confidence',
        nbins=20,
        title="Confidence Score Distribution",
        labels={'confidence': 'Confidence Score', 'count': 'Number of Tickets'}
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_confidence_by_category_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a box plot of confidence scores by category.
    
    Args:
        df: DataFrame with predictions and confidence scores
        
    Returns:
        Plotly figure
    """
    fig = px.box(
        df,
        x='predicted_category',
        y='confidence',
        title="Confidence Scores by Category"
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=500)
    
    return fig

def create_metrics_cards(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate key metrics for display.
    
    Args:
        df: DataFrame with predictions
        
    Returns:
        Dictionary of metrics
    """
    total_tickets = len(df)
    avg_confidence = df['confidence'].mean()
    high_confidence = len(df[df['confidence'] >= 0.8])
    low_confidence = len(df[df['confidence'] < 0.5])
    most_common_category = df['predicted_category'].value_counts().index[0]
    
    return {
        'total_tickets': total_tickets,
        'avg_confidence': avg_confidence,
        'high_confidence': high_confidence,
        'high_confidence_pct': (high_confidence / total_tickets) * 100,
        'low_confidence': low_confidence,
        'low_confidence_pct': (low_confidence / total_tickets) * 100,
        'most_common_category': most_common_category
    }

def display_metrics_dashboard(df: pd.DataFrame):
    """
    Display a metrics dashboard using Streamlit.
    
    Args:
        df: DataFrame with predictions
    """
    metrics = create_metrics_cards(df)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Tickets",
            f"{metrics['total_tickets']:,}",
            help="Total number of processed tickets"
        )
    
    with col2:
        st.metric(
            "Average Confidence",
            f"{metrics['avg_confidence']:.1%}",
            help="Average confidence score across all predictions"
        )
    
    with col3:
        st.metric(
            "High Confidence",
            f"{metrics['high_confidence']} ({metrics['high_confidence_pct']:.1f}%)",
            help="Tickets with confidence >= 80%"
        )
    
    with col4:
        st.metric(
            "Low Confidence",
            f"{metrics['low_confidence']} ({metrics['low_confidence_pct']:.1f}%)",
            help="Tickets with confidence < 50%"
        )

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interactive filters to a dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Filtered dataframe
    """
    with st.expander("🔍 Data Filters"):
        # Category filter
        if 'predicted_category' in df.columns:
            categories = ['All'] + sorted(df['predicted_category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category", categories)
            
            if selected_category != 'All':
                df = df[df['predicted_category'] == selected_category]
        
        # Confidence filter
        if 'confidence' in df.columns:
            min_confidence = float(df['confidence'].min())
            max_confidence = float(df['confidence'].max())
            
            confidence_range = st.slider(
                "Confidence Range",
                min_value=min_confidence,
                max_value=max_confidence,
                value=(min_confidence, max_confidence),
                step=0.01
            )
            
            df = df[
                (df['confidence'] >= confidence_range[0]) &
                (df['confidence'] <= confidence_range[1])
            ]
        
        # Priority filter (if available)
        if 'priority' in df.columns:
            priorities = ['All'] + sorted(df['priority'].unique().tolist())
            selected_priority = st.selectbox("Filter by Priority", priorities)
            
            if selected_priority != 'All':
                df = df[df['priority'] == selected_priority]
        
        # Channel filter (if available)
        if 'channel' in df.columns:
            channels = ['All'] + sorted(df['channel'].unique().tolist())
            selected_channel = st.selectbox("Filter by Channel", channels)
            
            if selected_channel != 'All':
                df = df[df['channel'] == selected_channel]
    
    return df

def create_time_series_chart(df: pd.DataFrame, date_column: str = 'timestamp') -> go.Figure:
    """
    Create a time series chart of ticket volume.
    
    Args:
        df: DataFrame with timestamp data
        date_column: Name of the date column
        
    Returns:
        Plotly figure
    """
    if date_column not in df.columns:
        return None
    
    # Convert to datetime if not already
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by date
    daily_counts = df.groupby(df[date_column].dt.date).size().reset_index()
    daily_counts.columns = ['date', 'count']
    
    fig = px.line(
        daily_counts,
        x='date',
        y='count',
        title="Ticket Volume Over Time",
        labels={'count': 'Number of Tickets', 'date': 'Date'}
    )
    
    fig.update_layout(height=400)
    
    return fig

def export_data_button(df: pd.DataFrame, filename: str = "data_export"):
    """
    Create a download button for exporting data.
    
    Args:
        df: DataFrame to export
        filename: Base filename for export
    """
    # CSV export
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv"
    )

def display_prediction_details(result: Dict[str, Any]):
    """
    Display detailed prediction results.
    
    Args:
        result: Prediction result dictionary
    """
    # Main prediction
    st.success(f"**Predicted Category:** {result['predicted_category']}")
    st.info(f"**Confidence:** {result['confidence']:.1%}")
    
    # Show probabilities if available
    if 'probabilities' in result:
        st.subheader("Category Probabilities")
        
        prob_df = pd.DataFrame([
            {'Category': cat, 'Probability': prob}
            for cat, prob in result['probabilities'].items()
        ]).sort_values('Probability', ascending=False)
        
        # Create horizontal bar chart
        fig = px.bar(
            prob_df,
            x='Probability',
            y='Category',
            orientation='h',
            title="Probability Distribution",
            color='Probability',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show as table
        st.dataframe(prob_df, use_container_width=True)

def validate_csv_format(df: pd.DataFrame, required_columns: List[str]) -> tuple:
    """
    Validate CSV format and required columns.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for empty required columns
    for col in required_columns:
        if df[col].isna().all():
            return False, f"Column '{col}' is completely empty"
    
    return True, "Format is valid"

def create_summary_report(df: pd.DataFrame) -> str:
    """
    Create a text summary report of the data.
    
    Args:
        df: DataFrame with prediction results
        
    Returns:
        Summary report as string
    """
    metrics = create_metrics_cards(df)
    
    report = f"""
# Ticket Classification Summary Report

## Overview
- **Total Tickets Processed**: {metrics['total_tickets']:,}
- **Average Confidence**: {metrics['avg_confidence']:.1%}
- **Most Common Category**: {metrics['most_common_category']}

## Confidence Distribution
- **High Confidence (≥80%)**: {metrics['high_confidence']} tickets ({metrics['high_confidence_pct']:.1f}%)
- **Low Confidence (<50%)**: {metrics['low_confidence']} tickets ({metrics['low_confidence_pct']:.1f}%)

## Category Breakdown
"""
    
    if 'predicted_category' in df.columns:
        category_counts = df['predicted_category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            report += f"- **{category}**: {count} tickets ({percentage:.1f}%)\n"
    
    if 'confidence' in df.columns:
        avg_conf_by_cat = df.groupby('predicted_category')['confidence'].mean()
        report += "\n## Average Confidence by Category\n"
        for category, conf in avg_conf_by_cat.items():
            report += f"- **{category}**: {conf:.1%}\n"
    
    return report
