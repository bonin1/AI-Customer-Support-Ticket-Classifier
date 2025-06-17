# AI Customer Support Ticket Classifier

An intelligent system that automatically classifies customer support tickets into predefined categories to speed up triage and routing processes.

## 🎯 Features

- **Automated Classification**: Classifies tickets into categories like Billing, Technical Issue, Feature Request, etc.
- **Multiple Model Support**: LSTM, BERT, and traditional ML models
- **Interactive Dashboard**: Streamlit-based UI with real-time predictions
- **Data Analysis**: Comprehensive ticket analytics and trends
- **Human-in-the-Loop**: Manual correction capabilities with model retraining
- **Batch Processing**: Upload CSV files for bulk classification

## 🔧 Technologies Used

- **NLP**: spaCy, NLTK, transformers for text processing
- **Machine Learning**: TensorFlow/Keras for deep learning models
- **Data Analysis**: Pandas, NumPy for data manipulation
- **Visualization**: Plotly, Matplotlib, Seaborn
- **UI**: Streamlit for interactive web interface

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Raw ticket data
│   ├── processed/              # Cleaned and preprocessed data
│   └── sample/                 # Sample datasets
├── models/
│   ├── saved_models/           # Trained model files
│   └── model_training.py       # Model training scripts
├── src/
│   ├── data_preprocessing.py   # Text cleaning and preprocessing
│   ├── model_builder.py        # Model architecture definitions
│   ├── train.py               # Training pipeline
│   └── predict.py             # Prediction utilities
├── streamlit_app/
│   ├── app.py                 # Main Streamlit application
│   ├── components/            # UI components
│   └── utils/                 # Helper functions
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data analysis notebooks
└── requirements.txt
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Prepare Data**
   - Place your ticket data in CSV format in `data/raw/`
   - Required columns: `ticket_id`, `customer_message`, `category`
   - Optional: `timestamp`, `priority`, `channel`

3. **Train Model**
   ```bash
   python src/train.py --data_path data/raw/tickets.csv --model_type lstm
   ```

4. **Run Streamlit App**
   ```bash
   streamlit run streamlit_app/app.py
   ```

## 📊 Supported Categories

- Billing & Payment
- Technical Issue
- Feature Request
- Account Management
- Product Information
- Refund & Return
- General Inquiry

## 🔄 Workflow

1. **Data Collection**: Load ticket data from CSV/JSON
2. **Preprocessing**: Clean text, tokenize, vectorize
3. **Model Training**: Train classification models
4. **Evaluation**: Assess model performance
5. **Deployment**: Use Streamlit interface for predictions
6. **Feedback Loop**: Collect corrections and retrain

## 📈 Model Performance

The system supports multiple model architectures:
- **LSTM**: Fast training, good baseline performance
- **BERT**: State-of-the-art accuracy for text classification
- **Traditional ML**: SVM, Random Forest for comparison

## 🤝 Human-in-the-Loop

- Manual correction interface
- Confidence score display
- Batch correction capabilities
- Automatic model retraining with corrected data

## 📊 Analytics Dashboard

- Ticket volume trends
- Category distribution
- Performance metrics
- Confidence score analysis

## 🛠️ Advanced Features

- **Custom Categories**: Easy to add new ticket categories
- **Multi-language Support**: Extensible for different languages
- **API Integration**: RESTful API for external systems
- **Automated Retraining**: Scheduled model updates
.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request


## 🆘 Support

For support, please open an issue.