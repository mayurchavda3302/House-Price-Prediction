import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(filepath):
    """Load and return the dataset with additional features."""
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Check for required columns
    required_columns = ['price', 'size_sqft', 'bedrooms', 'city', 'location']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Ensure numeric columns are of correct type
    numeric_cols = ['price', 'size_sqft', 'bedrooms']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing values in numeric columns
    df = df.dropna(subset=numeric_cols)
    
    # Feature Engineering
    df['price_per_sqft'] = df['price'] / df['size_sqft']
    df['bed_size_ratio'] = df['bedrooms'] / df['size_sqft']
    
    # Create a binary feature for metro city
    metro_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad']
    df['is_metro'] = df['city'].apply(lambda x: 1 if x in metro_cities else 0)
    
    return df

def explore_data(df):
    """Perform exploratory data analysis with visualizations."""
    print("\n=== Dataset Overview ===")
    print(f"Number of samples: {len(df)}")
    print("\nFirst few rows:")
    print(df.head().to_string())
    
    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    print("\nFeature correlation heatmap saved as 'feature_correlation.png'")
    
    # Price distribution by city
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='city', y='price', data=df)
    plt.title('Price Distribution by City')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_by_city.png')
    print("Price distribution by city saved as 'price_by_city.png'")
    
    return df

def build_advanced_model():
    """Build an advanced model with feature engineering and hyperparameter tuning."""
    # Define features
    numeric_features = ['size_sqft', 'bedrooms', 'price_per_sqft', 'bed_size_ratio', 'is_metro']
    categorical_features = ['city', 'location']
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression, k=10)),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Hyperparameter grid
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 5, 10],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, 
        param_grid=param_grid, 
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    return grid_search

def evaluate_model_improved(model, X_test, y_test):
    """Enhanced model evaluation with multiple metrics and visualizations."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # as percentage
    
    print("\n=== Model Evaluation ===")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: ₹{mae:,.2f}")
    print(f"Mean Squared Error: {mse:,.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Feature importance
    if hasattr(model.best_estimator_.named_steps['regressor'], 'feature_importances_'):
        feature_importances = model.best_estimator_.named_steps['regressor'].feature_importances_
        feature_names = model.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
        
        # Get selected features if feature selection was used
        if 'feature_selection' in model.best_estimator_.named_steps:
            selected_indices = model.best_estimator_.named_steps['feature_selection'].get_support(indices=True)
            feature_importances = feature_importances[selected_indices]
            feature_names = [feature_names[i] for i in selected_indices]
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
        plt.title('Top 10 Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("\nFeature importance plot saved as 'feature_importance.png'")
    
    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig('residual_plot.png')
    print("Residual plot saved as 'residual_plot.png'")
    
    return {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'mape': mape
    }

def main():
    print("=== Starting Improved House Price Prediction Model ===\n")
    
    # Load and explore data
    print("Loading and preparing data...")
    df = load_data('indian_property_sample.csv')
    df = explore_data(df)
    
    # Prepare data for modeling
    X = df.drop(['price', 'price_per_sqft'], axis=1)  # Remove target and derived features
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build and train model
    print("\n=== Training Improved Model ===")
    model = build_advanced_model()
    
    print("Performing grid search with cross-validation...")
    model.fit(X_train, y_train)
    
    print("\n=== Best Parameters ===")
    print(model.best_params_)
    
    # Evaluate model
    metrics = evaluate_model_improved(model, X_test, y_test)
    
    # Save the model
    joblib.dump(model.best_estimator_, 'improved_house_price_model.joblib')
    print("\n=== Model saved as 'improved_house_price_model.joblib' ===")
    
    # Example predictions
    print("\n=== Example Predictions ===")
    def predict_example(size_sqft, bedrooms, city, location):
        sample = pd.DataFrame({
            'size_sqft': [size_sqft],
            'bedrooms': [bedrooms],
            'city': [city],
            'location': [location],
            'price_per_sqft': 0,  # Will be calculated in the pipeline
            'bed_size_ratio': bedrooms / size_sqft,
            'is_metro': 1 if city in ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad'] else 0
        })
        
        # Make prediction
        prediction = model.predict(sample)[0]
        print(f"{bedrooms}BHK in {location}, {city} ({size_sqft} sqft): ₹{prediction:,.2f}")
        return prediction
    
    # Example predictions
    predict_example(1000, 2, 'Mumbai', 'Bandra')
    predict_example(1200, 3, 'Bangalore', 'Koramangala')
    predict_example(800, 1, 'Pune', 'Hinjewadi')

if __name__ == "__main__":
    main()
