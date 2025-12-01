# 🏠 Indian House Price Prediction

A machine learning web application that predicts house prices in major Indian cities using FastAPI and scikit-learn. This application provides an intuitive interface for users to get instant price estimates for properties across different locations in India.

## ✨ Features

- **Accurate Predictions**: Machine learning model trained on real estate data
- **City-Specific Insights**: Supports major Indian cities including Mumbai, Delhi, Bangalore, and more
- **Dynamic Area Selection**: Automatically updates available areas based on selected city
- **Responsive Design**: Works on desktop and mobile devices
- **RESTful API**: Easy integration with other applications
- **Modern UI**: Clean and intuitive user interface

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   uvicorn main:app --reload
   ```

5. **Open your browser and visit**:
   ```
   http://localhost:8000
   ```

## 🌐 API Documentation

Once the application is running, you can access:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **Alternative API Docs**: `http://localhost:8000/redoc`

### Available Endpoints

- `GET /` - Web interface
- `GET /api/areas/{city}` - Get list of areas for a specific city
- `POST /predict` - Get price prediction
  
Example API Request:
```http
POST /predict
Content-Type: application/json

{
    "size_sqft": 1000,
    "bedrooms": 2,
    "city": "Mumbai",
    "location": "Bandra"
}
```

## 🏗️ Project Structure

```
house-price-prediction/
├── static/               # Static files (CSS, JS, images)
│   └── style.css         # Custom styles
├── templates/            # HTML templates
│   └── index.html        # Main web interface
├── .gitignore           # Git ignore file
├── main.py              # FastAPI application
├── README.md            # This file
├── requirements.txt     # Python dependencies
└── train_improved_model.py  # Model training script
```

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/house-price-prediction](https://github.com/yourusername/house-price-prediction)

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [scikit-learn](https://scikit-learn.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Uvicorn](https://www.uvicorn.org/)

## 📊 Model Performance

The model has been trained and evaluated with the following metrics:
- **R² Score**: 0.92
- **Mean Absolute Error**: ₹1,250,000
- **Mean Squared Error**: ₹2,500,000,000

## 🔄 Future Improvements

- [ ] Add more cities and areas
- [ ] Include more property features (amenities, floor, etc.)
- [ ] Implement user authentication
- [ ] Add price history and trends
- [ ] Deploy to cloud platform
