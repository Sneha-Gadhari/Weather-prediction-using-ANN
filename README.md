# Weather Prediction and Air Quality Index (AQI) System

## Overview
This project is a machine learning-based web application that predicts weather conditions and air quality levels. It uses Artificial Neural Networks (ANN) to classify weather and pollution categories based on input data.

The system is built using Flask for the backend and provides an interactive UI for user input and visualization.

## Features
- Weather prediction using trained ANN model
- Air Quality Index (AQI) classification
- Interactive web interface
- Pre-trained models for fast predictions
- JSON-based API responses
- Visualization of prediction confidence

## Technologies Used
- Python
- Flask
- NumPy
- Pickle
- HTML (Jinja Templates)

## Project Structure
Weather prediction /
│

├── app.py # Main Flask application

├── train_pollution.py # Script to train pollution model

├── mumbai_data.py # Dataset preprocessing or handling

├── mumbai_aqi_dataset.csv # Dataset used for AQI prediction

├── best_ann_model.pkl # Trained weather prediction model

├── best_pollution_model.pkl # Trained pollution model

├── requirements.txt # Required dependencies

└── templates/

  └── index.html # Frontend UI


## Installation

### 1. Clone the Repository

git clone https://github.com/your-username/weather-aqi-prediction.git

cd weather-aqi-prediction


### 2. Install Dependencies

pip install -r requirements.txt


### 3. Run the Application

python app.py


### 4. Open in Browser

http://127.0.0.1:5000/


## How It Works
- The application loads pre-trained ANN models using pickle.
- User inputs are processed and converted into numerical features.
- The model predicts:
  - Weather category (Sunny, Rain, Storm, etc.)
  - AQI level (Good, Moderate, Unhealthy, Hazardous)
- Results are displayed with probability scores.

## Models Used
- Weather Model: ANN classifier trained on weather conditions
- Pollution Model: ANN classifier trained on AQI dataset

## Output Classes

### Weather Categories
- Sunny
- Cloudy
- Humid
- Rain
- Storm

### AQI Categories
- Good
- Moderate
- Unhealthy
- Hazardous

## Dataset
- Mumbai AQI dataset is used for pollution prediction
- Includes environmental and atmospheric parameters

## Future Improvements
- Integration with live weather APIs
- Real-time AQI updates
- Enhanced UI with charts and graphs
- Deployment on cloud platforms

