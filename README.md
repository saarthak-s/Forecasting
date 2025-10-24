# 📈 Time Series Forecasting Project

A comprehensive time series forecasting project focused on **energy load prediction** using multiple statistical and machine learning approaches. This project implements various forecasting models including SARIMA, Linear Regression, and weather-based predictions for Delhi's power load data from 2017-2024.

## 🎯 Project Overview

This project analyzes and forecasts electrical power load data for Delhi using high-frequency (15-minute interval) time series data spanning from April 2017 to June 2024. The implementation includes multiple forecasting approaches with weather data integration for improved prediction accuracy.

## ✨ Key Features

### 📊 **Multiple Forecasting Models**
- **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** - Statistical time series modeling
- **Multiple Linear Regression (MLR)** - Weather-integrated forecasting
- **Intra-day Forecasting** - Short-term prediction capabilities
- **Wind-based Models** - Weather factor incorporation

### 🔍 **Advanced Analysis Tools**
- **Time Series Decomposition** - Trend, seasonality, and residual analysis
- **Stationarity Testing** - ADF and KPSS tests
- **ACF/PACF Analysis** - Autocorrelation and partial autocorrelation
- **Model Validation** - MAPE (Mean Absolute Percentage Error) evaluation

### 🌤️ **Weather Integration**
- Temperature, humidity, and wind speed correlation
- Weather data preprocessing and resampling
- Multivariate forecasting with meteorological variables

### 📈 **Data Processing Capabilities**
- 15-minute interval time series handling
- Missing data interpolation
- Seasonal pattern recognition
- Multiple seasonality detection (daily, weekly, yearly)

## 🛠️ Tech Stack

### **Core Libraries**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization and plotting
- **statsmodels** - Statistical modeling and time series analysis
- **scikit-learn** - Machine learning algorithms
- **joblib** - Model persistence and parallel processing

### **Specialized Packages**
- **SARIMAX** - Seasonal ARIMA with exogenous variables
- **pmdarima** - Automated ARIMA model selection
- **MSTL** - Multiple Seasonal-Trend decomposition

## 📁 Project Structure

```
Forecasting/
├── weather.py              # Weather-based SARIMA forecasting
├── weather2.py             # Alternative weather model implementation
├── mlr.py                  # Multiple Linear Regression with weather data
├── wind.py                 # Wind-integrated forecasting models
├── intera-day.py           # Intra-day forecasting (short-term predictions)
├── error testing.py        # Model validation and error analysis
├── Plots.py                # Visualization and statistical analysis tools
├── Preprocess.py           # Data preprocessing and cleaning utilities
├── SARIMA Errors           # Model performance results and benchmarks
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib statsmodels scikit-learn joblib pmdarima
```

### Data Requirements
The project expects the following data structure:
- **Main Dataset**: `processed.xlsx` - 15-minute interval load data (2017-2024)
- **Weather Data**: `weather_may_june.csv` - Meteorological data
- **Frequency**: 15-minute intervals
- **Date Format**: `DD/MM/YYYY HH:MM`

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/saarthak-s/Forecasting.git
   cd Forecasting
   ```

2. **Prepare your data**
   - Place your load data in the expected format
   - Ensure weather data is available for multivariate models
   - Update file paths in the scripts to match your data location

3. **Run forecasting models**
   ```bash
   # SARIMA with weather data
   python weather.py
   
   # Multiple Linear Regression
   python mlr.py
   
   # Intra-day forecasting
   python intera-day.py
   
   # Statistical analysis and plots
   python Plots.py
   ```

## 📊 Model Performance

### Best SARIMA Model Results (June 2024)
```
Model: SARIMAX (2, 0, 0)x(1, 1, 0, 96)
MAPE: 3.6778%
```

### Model Comparison
| Model Configuration | MAPE | Performance |
|-------------------|------|-------------|
| SARIMAX (2,0,0)x(1,1,0,96) | 3.68% | ⭐ Best |
| SARIMAX (1,0,0)x(1,1,0,96) | 3.76% | ⭐ Good |
| SARIMAX (4,0,0)x(1,1,0,96) | 3.69% | ⭐ Good |
| Weather-MLR Model | Variable | 📊 Weather-dependent |

## 🔧 Key Functionalities

### 1. **SARIMA Forecasting** (`weather.py`, `weather2.py`)
- Seasonal pattern recognition (96 periods = daily seasonality with 15-min intervals)
- Automatic parameter optimization
- Weather variable integration
- Rolling forecast validation

### 2. **Linear Regression Models** (`mlr.py`)
- Weather feature engineering
- Temperature, humidity correlation analysis
- Multivariate time series regression
- Feature importance analysis

### 3. **Statistical Analysis** (`Plots.py`)
- **Stationarity Tests**: ADF and KPSS
- **Decomposition**: Trend, seasonal, residual components
- **Autocorrelation**: ACF/PACF plots for model identification
- **Visualization**: Time series plots and statistical diagnostics

### 4. **Data Processing** (`Preprocess.py`)
- Time series resampling and interpolation
- Missing data handling
- Date-time index management
- Data quality validation

### 5. **Model Validation** (`error testing.py`)
- Cross-validation frameworks
- MAPE calculation and comparison
- Residual analysis
- Forecast accuracy metrics

## 📈 Use Cases

### **Energy Sector Applications**
- **Load Forecasting**: Power grid management and planning
- **Demand Response**: Peak load prediction for pricing
- **Grid Stability**: Short-term load balancing
- **Renewable Integration**: Weather-dependent generation forecasting

### **Research Applications**
- Time series methodology comparison
- Weather impact quantification
- Seasonal pattern analysis
- High-frequency forecasting studies

## 🔍 Advanced Features

### **Multi-horizon Forecasting**
- Intra-day: 1-24 hours ahead
- Daily: 1-7 days ahead
- Weekly: 1-4 weeks ahead

### **Weather Integration**
- Temperature correlation analysis
- Humidity impact assessment
- Wind speed factor incorporation
- Seasonal weather pattern recognition

### **Model Ensemble**
- Multiple model averaging
- Weather-conditional model selection
- Forecast combination techniques

## 📊 Data Characteristics

- **Frequency**: 15-minute intervals (96 observations/day)
- **Seasonality**: Multiple (daily, weekly, annual)
- **Trend**: Long-term growth patterns
- **External Factors**: Weather variables, holidays
- **Data Quality**: Interpolated missing values, outlier handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Implement your forecasting model or improvement
4. Add appropriate documentation and tests
5. Submit a pull request

### Contribution Areas
- New forecasting algorithms
- Improved data preprocessing
- Advanced visualization tools
- Model evaluation metrics
- Performance optimizations

## 📄 License

This project is open source. Please see the repository for license details.

## 👥 Author

**Saarthak** - [saarthak-s](https://github.com/saarthak-s)

## 🙏 Acknowledgments

- **Delhi Load Data**: Historical power consumption data (2017-2024)
- **Weather Data**: Meteorological observations for forecasting enhancement
- **Statistical Libraries**: statsmodels, scikit-learn, pandas ecosystem
- **Research Community**: Time series forecasting methodologies

## 📞 Support

For questions, suggestions, or collaboration opportunities:
- Create an issue in the GitHub repository
- Email: [Your contact information]

---

**⚡ Forecasting the Future of Energy Management** 📊

*This project demonstrates the power of combining statistical modeling with weather data for accurate energy load forecasting, supporting smart grid initiatives and sustainable energy management.*