"""
NeuralForecast Demo Script
==========================
This script demonstrates the usage of the NeuralForecast library for time series forecasting
using state-of-the-art deep learning models.

Author: Dr. Mohammed Elseidi
Reference: https://github.com/Nixtla/neuralforecast
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, MLP, LSTM, TFT
from neuralforecast.utils import AirPassengersDF
from neuralforecast.losses.pytorch import MAE, MSE


def example_basic_forecast():
    """
    Basic example using NBEATS model with AirPassengers dataset.
    """
    print("=" * 60)
    print("Example 1: Basic NBEATS Forecast")
    print("=" * 60)

    # Load sample data
    df = AirPassengersDF
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

    # Initialize model
    horizon = 12  # Forecast 12 periods ahead

    nf = NeuralForecast(
        models=[
            NBEATS(
                input_size=24,      # Use 24 historical observations
                h=horizon,          # Forecast horizon
                max_steps=100,      # Training steps
                scaler_type='standard'
            )
        ],
        freq='ME'  # Monthly End frequency
    )

    # Fit and predict
    nf.fit(df=df)
    forecast = nf.predict()

    print(f"\nForecast:\n{forecast}")
    return forecast


def example_multiple_models():
    """
    Compare multiple neural network models on the same dataset.
    """
    print("\n" + "=" * 60)
    print("Example 2: Comparing Multiple Models")
    print("=" * 60)

    df = AirPassengersDF
    horizon = 12

    # Define multiple models
    models = [
        NBEATS(input_size=24, h=horizon, max_steps=50),
        NHITS(input_size=24, h=horizon, max_steps=50),
        MLP(input_size=24, h=horizon, max_steps=50, hidden_size=64),
    ]

    nf = NeuralForecast(models=models, freq='ME')
    nf.fit(df=df)
    forecasts = nf.predict()

    print(f"\nForecasts from multiple models:\n{forecasts}")
    return forecasts


def example_cross_validation():
    """
    Perform cross-validation to evaluate model performance.
    """
    print("\n" + "=" * 60)
    print("Example 3: Cross-Validation")
    print("=" * 60)

    df = AirPassengersDF
    horizon = 12

    nf = NeuralForecast(
        models=[NHITS(input_size=24, h=horizon, max_steps=50)],
        freq='ME'
    )

    # Cross-validation with 2 windows
    cv_results = nf.cross_validation(df=df, n_windows=2)

    print(f"\nCross-validation results:\n{cv_results}")

    # Calculate metrics
    mae = np.mean(np.abs(cv_results['y'] - cv_results['NHITS']))
    print(f"\nMean Absolute Error: {mae:.2f}")

    return cv_results


def example_probabilistic_forecast():
    """
    Generate probabilistic forecasts with prediction intervals.
    """
    print("\n" + "=" * 60)
    print("Example 4: Probabilistic Forecasting")
    print("=" * 60)

    df = AirPassengersDF
    horizon = 12

    # Use NHITS with quantile loss for probabilistic forecasts
    from neuralforecast.losses.pytorch import DistributionLoss

    nf = NeuralForecast(
        models=[
            NHITS(
                input_size=24,
                h=horizon,
                max_steps=50,
                loss=DistributionLoss(distribution='Normal'),
                valid_loss=MAE()
            )
        ],
        freq='ME'
    )

    nf.fit(df=df)
    forecasts = nf.predict()

    print(f"\nProbabilistic forecasts:\n{forecasts}")
    return forecasts


def example_custom_dataset():
    """
    Create and forecast on a custom synthetic dataset.
    Useful for testing on temperature or wind speed data.
    """
    print("\n" + "=" * 60)
    print("Example 5: Custom Synthetic Dataset (Temperature-like)")
    print("=" * 60)

    # Create synthetic temperature-like data with trend and seasonality
    np.random.seed(42)
    n_points = 365 * 3  # 3 years of daily data

    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')

    # Simulate temperature: trend + annual seasonality + noise
    trend = np.linspace(0, 2, n_points)  # Slight warming trend
    seasonal = 15 * np.sin(2 * np.pi * np.arange(n_points) / 365)  # Annual cycle
    noise = np.random.normal(0, 3, n_points)
    temperature = 20 + trend + seasonal + noise  # Base temp around 20C

    df = pd.DataFrame({
        'unique_id': 'temperature_station_1',
        'ds': dates,
        'y': temperature
    })

    print(f"Custom dataset shape: {df.shape}")
    print(f"Temperature statistics:")
    print(f"  Mean: {df['y'].mean():.2f}C")
    print(f"  Std: {df['y'].std():.2f}C")
    print(f"  Min: {df['y'].min():.2f}C")
    print(f"  Max: {df['y'].max():.2f}C")

    # Forecast next 30 days
    horizon = 30

    nf = NeuralForecast(
        models=[
            NHITS(
                input_size=60,      # Use 60 days of history
                h=horizon,
                max_steps=100,
                n_pool_kernel_size=[2, 2, 2],
                n_freq_downsample=[4, 2, 1]
            )
        ],
        freq='D'
    )

    nf.fit(df=df)
    forecasts = nf.predict()

    print(f"\nTemperature forecast for next {horizon} days:\n{forecasts}")
    return forecasts, df


def plot_forecast(historical_df, forecast_df, model_name='NHITS', title='Forecast'):
    """
    Plot historical data and forecasts.
    """
    plt.figure(figsize=(12, 6))

    # Plot last 90 days of historical data
    recent_history = historical_df.tail(90)
    plt.plot(recent_history['ds'], recent_history['y'],
             label='Historical', color='blue', linewidth=2)

    # Plot forecast
    plt.plot(forecast_df['ds'], forecast_df[model_name],
             label='Forecast', color='red', linewidth=2, linestyle='--')

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('forecast_plot.png', dpi=150)
    plt.close()
    print("Plot saved to forecast_plot.png")


def main():
    """
    Run all examples.
    """
    print("NeuralForecast Demonstration")
    print("=" * 60)
    print("Library for state-of-the-art neural time series forecasting")
    print("GitHub: https://github.com/Nixtla/neuralforecast")
    print("=" * 60)

    # Run examples
    try:
        example_basic_forecast()
    except Exception as e:
        print(f"Example 1 error: {e}")

    try:
        example_multiple_models()
    except Exception as e:
        print(f"Example 2 error: {e}")

    try:
        example_cross_validation()
    except Exception as e:
        print(f"Example 3 error: {e}")

    try:
        example_probabilistic_forecast()
    except Exception as e:
        print(f"Example 4 error: {e}")

    try:
        forecasts, df = example_custom_dataset()
        plot_forecast(df, forecasts, model_name='NHITS',
                     title='Temperature Forecast (30 days)')
    except Exception as e:
        print(f"Example 5 error: {e}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
