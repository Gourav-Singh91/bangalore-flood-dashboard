# Bangalore Urban Flood Prediction Dashboard

A comprehensive dashboard for predicting and analyzing urban flooding in Bangalore using machine learning models. This project provides real-time flood predictions, historical analysis, and geographic visualization of flood-prone areas.

## 🌟 Features

- **Real-time Flood Prediction**: Input environmental parameters to get instant flood risk assessment
- **Multiple ML Models**: 
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - AdaBoost
  - SVM
  - KNN
  - Naive Bayes
- **Interactive Visualizations**:
  - Geographic distribution of flood events
  - Weather pattern analysis
  - Historical flood timeline
  - Feature importance analysis
  - Correlation matrix
- **Advanced Analytics**:
  - Risk scoring system
  - Model performance comparison
  - Cross-validation results
  - Ensemble model visualization

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/bangalore-flood-dashboard.git
cd bangalore-flood-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Dashboard

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## 📊 Data Sources

The dashboard uses the following data:
- Rainfall Intensity
- Temperature
- Humidity
- Atmospheric Pressure
- River Level
- Drainage Capacity
- Geographic coordinates (Latitude/Longitude)

## 🛠️ Technologies Used

- Python
- Streamlit
- Plotly
- Scikit-learn
- Pandas
- Folium
- XGBoost
- CatBoost
- LightGBM

## 📈 Model Performance

The dashboard includes multiple machine learning models with cross-validation and ensemble methods for improved accuracy. Each model's performance metrics are displayed in the dashboard for comparison.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- OpenWeatherMap API for weather data
- Bangalore Municipal Corporation for flood event data
- Contributors and maintainers of all used libraries 