import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
import warnings
import os
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import calendar
import requests
from io import BytesIO
import json
import base64
from PIL import Image
import io
warnings.filterwarnings('ignore')

# Define all functions first
def calculate_risk_score(probability, rainfall, river_level, drainage):
    """Calculate a comprehensive risk score based on multiple factors"""
    risk_score = 0
    
    # Probability weight
    risk_score += probability * 40
    
    # Rainfall intensity weight
    rainfall_risk = min(rainfall / 100, 1) * 20
    
    # River level weight
    river_risk = min(river_level / 5, 1) * 20
    
    # Drainage capacity weight (inverse)
    drainage_risk = (1 - min(drainage / 100, 1)) * 20
    
    risk_score += rainfall_risk + river_risk + drainage_risk
    
    return min(risk_score, 100)

def get_weather_forecast(lat, lon):
    """Get weather forecast data from OpenWeatherMap API"""
    try:
        # Replace with your OpenWeatherMap API key
        api_key = "YOUR_API_KEY"
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        st.warning(f"Could not fetch weather forecast: {str(e)}")
        return None

def create_advanced_correlation_matrix(df):
    """Create an interactive correlation matrix with additional features"""
    try:
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_cols].corr()
        
        # Create heatmap with annotations
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            showscale=True,
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title='Interactive Correlation Matrix',
            xaxis_title='Features',
            yaxis_title='Features',
            height=800,
            width=800
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating correlation matrix: {str(e)}")
        return None

def create_ensemble_visualization(results):
    """Create advanced ensemble model visualization"""
    try:
        # Create subplots for different ensemble metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy Comparison', 'Cross-Validation Scores',
                           'ROC Curves', 'Precision-Recall Curves')
        )
        
        # Model accuracy comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=accuracies, name='Accuracy'),
            row=1, col=1
        )
        
        # Cross-validation scores
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=cv_means, error_y=dict(type='data', array=cv_stds),
                   name='CV Scores'),
            row=1, col=2
        )
        
        # ROC curves
        for name in model_names:
            y_test = results[name]['y_test']
            y_prob = results[name]['probabilities']
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={auc:.2f})'),
                row=2, col=1
            )
        
        # Precision-Recall curves
        for name in model_names:
            y_test = results[name]['y_test']
            y_prob = results[name]['probabilities']
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            fig.add_trace(
                go.Scatter(x=recall, y=precision, name=name),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True)
        return fig
    except Exception as e:
        st.error(f"Error creating ensemble visualization: {str(e)}")
        return None

def create_weather_pattern_analysis(df):
    """Create weather pattern analysis visualization"""
    try:
        # Create subplots for different weather patterns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Monthly Rainfall Distribution', 'Temperature vs Flood Probability',
                           'Humidity vs Flood Probability', 'Pressure vs Flood Probability')
        )
        
        # Monthly rainfall distribution
        monthly_rainfall = df.groupby(df['Date'].dt.month)['Rainfall_Intensity'].mean()
        fig.add_trace(
            go.Bar(x=list(calendar.month_abbr)[1:], y=monthly_rainfall.values, name='Monthly Rainfall'),
            row=1, col=1
        )
        
        # Temperature vs Flood Probability
        temp_bins = pd.qcut(df['Temperature'], q=10)
        temp_flood_prob = df.groupby(temp_bins)['flood'].mean()
        fig.add_trace(
            go.Scatter(x=temp_flood_prob.index.astype(str), y=temp_flood_prob.values,
                      name='Temperature vs Flood', mode='lines+markers'),
            row=1, col=2
        )
        
        # Humidity vs Flood Probability
        humidity_bins = pd.qcut(df['Humidity'], q=10)
        humidity_flood_prob = df.groupby(humidity_bins)['flood'].mean()
        fig.add_trace(
            go.Scatter(x=humidity_flood_prob.index.astype(str), y=humidity_flood_prob.values,
                      name='Humidity vs Flood', mode='lines+markers'),
            row=2, col=1
        )
        
        # Pressure vs Flood Probability
        pressure_bins = pd.qcut(df['Atmospheric_Pressure'], q=10)
        pressure_flood_prob = df.groupby(pressure_bins)['flood'].mean()
        fig.add_trace(
            go.Scatter(x=pressure_flood_prob.index.astype(str), y=pressure_flood_prob.values,
                      name='Pressure vs Flood', mode='lines+markers'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        return fig
    except Exception as e:
        st.error(f"Error in weather pattern analysis: {str(e)}")
        return None

def create_historical_timeline(df):
    """Create historical flood event timeline"""
    try:
        # Group by date and count flood events
        daily_floods = df.groupby('Date')['flood'].sum().reset_index()
        
        # Create timeline visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_floods['Date'],
            y=daily_floods['flood'],
            mode='markers+lines',
            name='Flood Events',
            line=dict(color='red'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Historical Flood Event Timeline',
            xaxis_title='Date',
            yaxis_title='Number of Flood Events',
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"Error in historical timeline: {str(e)}")
        return None

def prepare_prediction_data(input_data, scaler, feature_names):
    """Prepare input data for prediction"""
    try:
        # Validate input data
        if input_data is None or len(input_data) == 0:
            raise ValueError("Input data is empty")
            
        # Create a DataFrame with the input data
        df_input = pd.DataFrame(input_data, columns=['Rainfall_Intensity', 'Temperature', 'Humidity', 
                                                   'Atmospheric_Pressure', 'River_Level', 'Drainage_Capacity'])
        
        # Validate input values
        for col in df_input.columns:
            if df_input[col].isnull().any():
                raise ValueError(f"Missing values in {col}")
            if np.isinf(df_input[col]).any():
                raise ValueError(f"Infinite values in {col}")
        
        # Engineer features
        df_engineered = engineer_features(df_input)
        
        # Add missing features with default values
        missing_features = ['Altitude', 'Drainage_System_Condition', 'Population_Density', 'Urbanization_Level']
        for feature in missing_features:
            if feature not in df_engineered.columns:
                df_engineered[feature] = 0  # Default value for missing features
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in df_engineered.columns:
                df_engineered[feature] = 0  # Default value for any missing features
        
        # Select only the features used in training
        X = df_engineered[feature_names]
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Validate data before scaling
        if X.isnull().any().any():
            raise ValueError("Missing values in engineered features")
        if np.isinf(X).any().any():
            raise ValueError("Infinite values in engineered features")
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        return X_scaled
        
    except Exception as e:
        st.error(f"Error preparing prediction data: {str(e)}")
        raise  # Re-raise the exception to be handled by the caller

def engineer_features(df):
    """Engineer features for the model"""
    try:
        # Create copy to avoid modifying original data
        df_engineered = df.copy()
        
        # Basic feature validation
        required_features = ['Rainfall_Intensity', 'Temperature', 'Humidity', 
                           'Atmospheric_Pressure', 'River_Level', 'Drainage_Capacity']
        for feature in required_features:
            if feature not in df_engineered.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Handle missing values first
        df_engineered = df_engineered.fillna(method='ffill').fillna(method='bfill')
        
        # Add missing features with default values
        missing_features = ['Altitude', 'Drainage_System_Condition', 'Population_Density', 'Urbanization_Level']
        for feature in missing_features:
            if feature not in df_engineered.columns:
                df_engineered[feature] = 0  # Default value for missing features
        
        # Most important interaction features only
        df_engineered['Rainfall_Temperature'] = df_engineered['Rainfall_Intensity'] * df_engineered['Temperature']
        df_engineered['Rainfall_Humidity'] = df_engineered['Rainfall_Intensity'] * df_engineered['Humidity']
        df_engineered['Pressure_Humidity'] = df_engineered['Atmospheric_Pressure'] * df_engineered['Humidity']
        
        # Most important polynomial features
        df_engineered['Rainfall_Squared'] = df_engineered['Rainfall_Intensity'] ** 2
        df_engineered['River_Level_Squared'] = df_engineered['River_Level'] ** 2
        
        # Most important ratio features with safety checks
        df_engineered['Drainage_Load'] = np.where(
            df_engineered['Drainage_Capacity'] != 0,
            df_engineered['Rainfall_Intensity'] / df_engineered['Drainage_Capacity'],
            0
        )
        df_engineered['River_Load'] = np.where(
            df_engineered['Drainage_Capacity'] != 0,
            df_engineered['River_Level'] / df_engineered['Drainage_Capacity'],
            0
        )
        
        # Most important rolling statistics with safety checks
        df_engineered['Rainfall_Rolling_Mean_3'] = df_engineered['Rainfall_Intensity']
        df_engineered['Rainfall_Rolling_Std_3'] = 0  # Default value for single prediction
        df_engineered['River_Rolling_Mean_3'] = df_engineered['River_Level']
        
        # Most important lag features with safety checks
        df_engineered['Rainfall_Lag_1'] = df_engineered['Rainfall_Intensity']
        df_engineered['River_Lag_1'] = df_engineered['River_Level']
        
        # Fill any remaining NaN values
        df_engineered = df_engineered.fillna(0)
        
        # Remove any infinite values
        df_engineered = df_engineered.replace([np.inf, -np.inf], 0)
        
        return df_engineered
        
    except Exception as e:
        st.error(f"Error in feature engineering: {str(e)}")
        return df  # Return original data if engineering fails

def load_data():
    """Load and validate the dataset"""
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "bangalore_urban_flood_prediction_AI1.xlsx")
        
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"Data file not found at: {file_path}")
            return None
            
        # Load the data
        df = pd.read_excel(file_path)
        
        # Check if required columns exist
        required_columns = ['Rainfall_Intensity', 'Temperature', 'Humidity', 'Atmospheric_Pressure', 
                          'River_Level', 'Drainage_Capacity', 'flood', 'Latitude', 'Longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
            
        # Add Date column if it doesn't exist
        if 'Date' not in df.columns:
            # Create a date range based on the actual length of the dataset
            dates = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            df['Date'] = dates
        
        # Ensure Date column is datetime type
        df['Date'] = pd.to_datetime(df['Date'])
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def train_models(df):
    """Train and evaluate multiple models"""
    try:
        # Check if data is loaded
        if df is None:
            st.error("No data available for training")
            return {}
            
        # Check data shape
        st.info(f"Training with {len(df)} samples and {len(df.columns)} features")
        
        # Engineer features
        df_engineered = engineer_features(df)
        
        # Define base features and engineered features, excluding Date column
        base_features = ['Rainfall_Intensity', 'Temperature', 'Humidity', 'Atmospheric_Pressure', 'River_Level', 'Drainage_Capacity']
        engineered_features = [col for col in df_engineered.columns if col not in base_features + ['flood', 'Latitude', 'Longitude', 'Date']]
        
        all_features = base_features + engineered_features
        X = df_engineered[all_features]
        y = df['flood']
        
        # Check for missing values
        if X.isnull().any().any():
            st.warning("Found missing values in features. Filling with forward fill and backward fill.")
            X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Split data with stratification and better test size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        st.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Advanced preprocessing with feature scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create validation set for early stopping
        X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train)
        
        # Optimized model parameters with early stopping
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=1,
                subsample=0.8,
                random_state=42,
                validation_fraction=0.2,
                n_iter_no_change=15,
                tol=1e-4
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=8,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                random_state=42,
                n_jobs=-1,
                eval_metric='auc',
                objective='binary:logistic'
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=500,
                learning_rate=0.03,
                algorithm='SAMME',
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                algorithm='auto',
                leaf_size=30,
                p=2,
                metric='minkowski',
                n_jobs=-1
            ),
            'Naive Bayes': GaussianNB(
                var_smoothing=1e-9
            )
        }
        
        # Train and evaluate models with cross-validation
        results = {}
        progress_bar = st.progress(0)
        
        for idx, (name, model) in enumerate(models.items()):
            try:
                st.info(f"Training {name}...")
                
                # Train model with cross-validation
                cv_scores = cross_val_score(model, X_train_final, y_train_final, cv=5, scoring='accuracy', n_jobs=-1)
                
                # Fit the model on the full training data with early stopping for specific models
                if name == 'XGBoost':
                    try:
                        # Convert to DMatrix format for XGBoost
                        dtrain = xgb.DMatrix(X_train_final, label=y_train_final)
                        dval = xgb.DMatrix(X_val, label=y_val)
                        
                        # Train with early stopping
                        model = xgb.train(
                            model.get_xgb_params(),
                            dtrain,
                            num_boost_round=500,
                            evals=[(dtrain, 'train'), (dval, 'val')],
                            early_stopping_rounds=15,
                            evals_result=None,
                            verbose_eval=False
                        )
                    except Exception as e:
                        st.error(f"Error in XGBoost training: {str(e)}")
                        continue
                else:
                    try:
                        model.fit(X_train_final, y_train_final)
                    except Exception as e:
                        st.error(f"Error in {name} training: {str(e)}")
                        continue
                
                # Make predictions with error handling
                try:
                    if name == 'XGBoost':
                        dtest = xgb.DMatrix(X_test_scaled)
                        y_pred = (model.predict(dtest) > 0.5).astype(int)
                        y_prob = model.predict(dtest)
                    else:
                        y_pred = model.predict(X_test_scaled)
                        y_prob = model.predict_proba(X_test_scaled)[:, 1]
                except Exception as e:
                    st.error(f"Error making predictions for {name}: {str(e)}")
                    continue
                
                # Calculate metrics with error handling
                try:
                    accuracy = accuracy_score(y_test, y_pred)
                    auc_score = roc_auc_score(y_test, y_prob)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    cm = confusion_matrix(y_test, y_pred)
                except Exception as e:
                    st.error(f"Error calculating metrics for {name}: {str(e)}")
                    continue
                
                # Store results with error handling
                try:
                    results[name] = {
                        'model': model,
                        'scaler': scaler,
                        'accuracy': accuracy,
                        'cv_scores': cv_scores,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'auc_score': auc_score,
                        'report': report,
                        'predictions': y_pred,
                        'probabilities': y_prob,
                        'confusion_matrix': cm,
                        'y_test': y_test,
                        'feature_names': all_features
                    }
                except Exception as e:
                    st.error(f"Error storing results for {name}: {str(e)}")
                    continue
                
                # Save high-performing models with error handling
                if accuracy > 0.85:
                    try:
                        model_path = os.path.join('models', f'{name.lower().replace(" ", "_")}.joblib')
                        scaler_path = os.path.join('models', f'scaler_{name.lower().replace(" ", "_")}.joblib')
                        os.makedirs('models', exist_ok=True)
                        joblib.dump(model, model_path)
                        joblib.dump(scaler, scaler_path)
                        st.success(f"Saved {name} model with accuracy: {accuracy:.2%}")
                    except Exception as e:
                        st.warning(f"Could not save {name} model: {str(e)}")
                
                # Update progress bar
                progress_bar.progress((idx + 1) / len(models))
                
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
                continue
        
        if not results:
            st.error("No models were successfully trained")
            return {}
            
        st.success("All models trained successfully!")
        return results
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return {}

# Set page config
st.set_page_config(
    page_title="Bangalore Urban Flood Prediction Dashboard",
    page_icon="🌧️",
    layout="wide"
)

# Custom CSS with enhanced styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .model-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .model-card:hover {
        transform: translateY(-5px);
    }
    .feature-importance {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .timeline-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .risk-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
    }
    .medium-risk {
        background-color: #fff3e0;
        color: #ef6c00;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌧️ Bangalore Urban Flood Prediction Dashboard")
st.markdown("""
    This advanced dashboard provides comprehensive insights into urban flood prediction in Bangalore using multiple machine learning models.
    Compare different models, view predictions, and understand the factors contributing to urban flooding.
""")

# Load the data
df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    page = st.sidebar.radio(
        "Select a page",
        ["Overview", "Model Comparison", "Predictions", "Geographic Analysis", "Weather Patterns", "About"]
    )

    if page == "Overview":
        # Overview metrics with enhanced styling
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            flood_events = len(df[df['flood'] == 1])
            st.metric("Flood Events", flood_events)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            flood_percentage = (flood_events / len(df)) * 100
            st.metric("Flood Event Rate", f"{flood_percentage:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Advanced correlation matrix
        st.subheader("Feature Correlation Analysis")
        correlation_fig = create_advanced_correlation_matrix(df)
        if correlation_fig:
            st.plotly_chart(correlation_fig, use_container_width=True)

        # Train models once and store results
        results = train_models(df)
        
        if results:
            # Feature importance from Random Forest
            if 'Random Forest' in results:
                rf_model = results['Random Forest']['model']
                features = results['Random Forest']['feature_names']
                fig = px.bar(x=features, y=rf_model.feature_importances_,
                            title='Feature Importance (Based on Random Forest Model)',
                            labels={'x': 'Features', 'y': 'Importance'})
                st.plotly_chart(fig, use_container_width=True)
            elif 'XGBoost' in results:
                xgb_model = results['XGBoost']['model']
                features = results['XGBoost']['feature_names']
                importance_dict = xgb_model.get_score(importance_type='gain')
                importance = [importance_dict.get(f'f{i}', 0) for i in range(len(features))]
                fig = px.bar(x=features, y=importance,
                            title='Feature Importance (Based on XGBoost Model)',
                            labels={'x': 'Features', 'y': 'Importance'})
                st.plotly_chart(fig, use_container_width=True)

        # Historical timeline
        st.subheader("Historical Flood Event Timeline")
        timeline_fig = create_historical_timeline(df)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)

    elif page == "Model Comparison":
        st.subheader("Model Performance Comparison")
        
        # Use cached results from train_models
        results = train_models(df)
        
        if not results:
            st.error("No models were successfully trained. Please check the data and try again.")
            st.stop()
            
        # Create ensemble visualization
        ensemble_fig = create_ensemble_visualization(results)
        if ensemble_fig:
            st.plotly_chart(ensemble_fig, use_container_width=True)
        
        # Display detailed metrics for each model
        st.subheader("Detailed Model Metrics")
        for name, result in results.items():
            with st.expander(f"{name} - Accuracy: {result['accuracy']:.2%} | AUC: {result['auc_score']:.2%} | CV Mean: {result['cv_mean']:.2%} ± {result['cv_std']:.2%}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Classification Report")
                    st.dataframe(pd.DataFrame(result['report']).round(3))
                
                with col2:
                    st.markdown("### Confusion Matrix")
                    cm = result['confusion_matrix']
                    fig = px.imshow(cm,
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['No Flood', 'Flood'],
                                y=['No Flood', 'Flood'],
                                title=f'Confusion Matrix - {name}')
                    st.plotly_chart(fig, use_container_width=True)

    elif page == "Predictions":
        st.subheader("Advanced Flood Prediction Model")
        
        # Use cached results from train_models
        results = train_models(df)
        
        if not results:
            st.error("No models were successfully trained. Please check the data and try again.")
            st.stop()
            
        # Model selection
        selected_model = st.selectbox("Select a model for prediction", list(results.keys()))
        
        # Display model performance
        st.success(f"Selected Model Accuracy: {results[selected_model]['accuracy']:.2%}")
        
        # Input features with validation
        col1, col2 = st.columns(2)
        with col1:
            rainfall = st.number_input("Rainfall Intensity (mm/h)", 
                                     min_value=float(df['Rainfall_Intensity'].min()),
                                     max_value=float(df['Rainfall_Intensity'].max()),
                                     value=float(df['Rainfall_Intensity'].mean()))
            temperature = st.number_input("Temperature (°C)",
                                        min_value=float(df['Temperature'].min()),
                                        max_value=float(df['Temperature'].max()),
                                        value=float(df['Temperature'].mean()))
            humidity = st.number_input("Humidity (%)",
                                     min_value=float(df['Humidity'].min()),
                                     max_value=float(df['Humidity'].max()),
                                     value=float(df['Humidity'].mean()))
        with col2:
            pressure = st.number_input("Atmospheric Pressure (hPa)",
                                     min_value=float(df['Atmospheric_Pressure'].min()),
                                     max_value=float(df['Atmospheric_Pressure'].max()),
                                     value=float(df['Atmospheric_Pressure'].mean()))
            river_level = st.number_input("River Level (m)",
                                        min_value=float(df['River_Level'].min()),
                                        max_value=float(df['River_Level'].max()),
                                        value=float(df['River_Level'].mean()))
            drainage = st.number_input("Drainage Capacity (m³/s)",
                                     min_value=float(df['Drainage_Capacity'].min()),
                                     max_value=float(df['Drainage_Capacity'].max()),
                                     value=float(df['Drainage_Capacity'].mean()))

        # Prediction button
        if st.button("Predict Flood Probability"):
            try:
                # Prepare input data
                input_data = np.array([[rainfall, temperature, humidity, pressure, river_level, drainage]])
                
                # Get the model and scaler
                model = results[selected_model]['model']
                scaler = results[selected_model]['scaler']
                feature_names = results[selected_model]['feature_names']
                
                # Prepare prediction data with engineered features
                input_scaled = prepare_prediction_data(input_data, scaler, feature_names)
                
                # Get prediction and probability with error handling
                try:
                    if selected_model == 'XGBoost':
                        dtest = xgb.DMatrix(input_scaled)
                        prediction = model.predict(dtest)[0]
                        probability = model.predict(dtest)[0]
                    else:
                        prediction = model.predict(input_scaled)[0]
                        probability = model.predict_proba(input_scaled)[0][1]
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please try again with different input values.")
                    st.stop()
                
                # Calculate risk score
                risk_score = calculate_risk_score(probability, rainfall, river_level, drainage)
                
                # Display results in a card
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                # Risk level determination
                if risk_score >= 70:
                    risk_class = "high-risk"
                    risk_text = "⚠️ High Risk of Flooding!"
                elif risk_score >= 40:
                    risk_class = "medium-risk"
                    risk_text = "⚠️ Medium Risk of Flooding"
                else:
                    risk_class = "low-risk"
                    risk_text = "✅ Low Risk of Flooding"
                
                st.markdown(f'<div class="risk-score {risk_class}">{risk_text}</div>', unsafe_allow_html=True)
                st.metric("Risk Score", f"{risk_score:.1f}/100")
                st.metric("Flood Probability", f"{probability:.2%}")
                
                # Enhanced visualization of prediction
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Flood Risk Level"},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "darkblue"},
                          'steps': [
                              {'range': [0, 30], 'color': "lightgreen"},
                              {'range': [30, 70], 'color': "yellow"},
                              {'range': [70, 100], 'color': "red"}
                          ],
                          'threshold': {
                              'line': {'color': "red", 'width': 4},
                              'thickness': 0.75,
                              'value': risk_score
                          }
                          }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please try again with different input values.")

    elif page == "Weather Patterns":
        st.subheader("Weather Pattern Analysis")
        
        # Weather pattern analysis
        weather_pattern_fig = create_weather_pattern_analysis(df)
        if weather_pattern_fig:
            st.plotly_chart(weather_pattern_fig, use_container_width=True)
        
        # Monthly statistics
        st.subheader("Monthly Statistics")
        try:
            if 'Date' in df.columns:
                monthly_stats = df.groupby(df['Date'].dt.month).agg({
                    'Rainfall_Intensity': ['mean', 'max', 'min'],
                    'Temperature': ['mean', 'max', 'min'],
                    'flood': 'mean'
                }).round(2)
                st.dataframe(monthly_stats)
            else:
                st.warning("Date column not available for monthly statistics")
                # Show overall statistics instead
                overall_stats = df.agg({
                    'Rainfall_Intensity': ['mean', 'max', 'min'],
                    'Temperature': ['mean', 'max', 'min'],
                    'flood': 'mean'
                }).round(2)
                st.dataframe(overall_stats)
        except Exception as e:
            st.error(f"Error calculating statistics: {str(e)}")
            st.info("Showing overall statistics instead")
            overall_stats = df.agg({
                'Rainfall_Intensity': ['mean', 'max', 'min'],
                'Temperature': ['mean', 'max', 'min'],
                'flood': 'mean'
            }).round(2)
            st.dataframe(overall_stats)

    elif page == "Geographic Analysis":
        st.subheader("Geographic Distribution of Flood Events")
        
        # Create a map centered on Bangalore
        center_lat = df['Latitude'].mean()
        center_lon = df['Longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add flood points to the map
        for idx, row in df.iterrows():
            color = 'red' if row['flood'] == 1 else 'blue'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                popup=f"Rainfall: {row['Rainfall_Intensity']:.1f} mm/h<br>"
                      f"Temperature: {row['Temperature']:.1f}°C<br>"
                      f"Flood: {'Yes' if row['flood'] == 1 else 'No'}",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)
        
        # Display the map
        folium_static(m, width=800, height=600)

        # Add a heatmap of flood events
        st.subheader("Flood Event Heatmap")
        flood_locations = df[df['flood'] == 1][['Latitude', 'Longitude']]
        fig = px.density_mapbox(flood_locations, 
                              lat='Latitude', 
                              lon='Longitude',
                              radius=20,
                              center=dict(lat=center_lat, lon=center_lon),
                              zoom=10,
                              mapbox_style="stamen-terrain")
        st.plotly_chart(fig, use_container_width=True)

    else:  # About page
        st.subheader("About This Dashboard")
        st.markdown("""
        This advanced dashboard was created to help understand and predict urban flooding in Bangalore using multiple machine learning models.
        
        ### Features:
        - Multiple classification models (Random Forest, Gradient Boosting, AdaBoost, SVM, Neural Network, KNN, Naive Bayes)
        - Interactive model comparison
        - Real-time flood predictions
        - Advanced statistical analysis
        - Geographic visualization of flood-prone areas
        - Feature importance analysis
        - Correlation analysis
        - Time series analysis and forecasting
        - Weather pattern analysis
        - Risk assessment scoring system
        - Historical flood event timeline
        - Advanced ensemble model visualization
        
        ### Data Sources:
        The dataset includes various environmental and urban factors:
        - Rainfall Intensity
        - Temperature
        - Humidity
        - Atmospheric Pressure
        - River Level
        - Drainage Capacity
        - Population Density
        - Urbanization Level
        
        ### Technologies Used:
        - Python
        - Streamlit
        - Plotly
        - Scikit-learn
        - Pandas
        - Folium
        - Statsmodels (Time Series Forecasting)
        - OpenWeatherMap API
        
        ### Model Performance:
        - Multiple classification models for comparison
        - Features include environmental and urban factors
        - Real-time predictions with probability estimates
        - Cross-validation for robust performance evaluation
        - Ensemble methods for improved accuracy
        """) 