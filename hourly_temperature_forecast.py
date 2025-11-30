"""
Hourly Temperature Forecasting with NeuralForecast
===================================================
This script demonstrates NeuralForecast applied to simulated hourly temperature data
with realistic patterns including daily and annual seasonality.

Author: Dr. Mohammed Elseidi
Reference: https://github.com/Nixtla/neuralforecast
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, TiDE, PatchTST
from neuralforecast.losses.pytorch import MAE, MSE, DistributionLoss


def generate_hourly_temperature_data(
    start_date='2022-01-01',
    n_days=365 * 2,
    base_temp=15.0,
    annual_amplitude=12.0,
    daily_amplitude=8.0,
    noise_std=2.0,
    seed=42
):
    """
    Generate realistic hourly temperature data with multiple seasonality patterns.

    Parameters:
    -----------
    start_date : str
        Start date for the time series
    n_days : int
        Number of days to simulate
    base_temp : float
        Base temperature in Celsius
    annual_amplitude : float
        Amplitude of annual seasonal variation
    daily_amplitude : float
        Amplitude of daily temperature variation
    noise_std : float
        Standard deviation of random noise
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['unique_id', 'ds', 'y'] ready for NeuralForecast
    """
    np.random.seed(seed)

    n_hours = n_days * 24
    dates = pd.date_range(start=start_date, periods=n_hours, freq='h')

    # Time indices
    hour_of_day = dates.hour
    day_of_year = dates.dayofyear

    # Annual seasonality (warmest in summer ~day 200, coldest in winter ~day 15)
    annual_seasonal = annual_amplitude * np.sin(2 * np.pi * (day_of_year - 105) / 365)

    # Daily seasonality (warmest at 14:00, coldest at 05:00)
    daily_seasonal = daily_amplitude * np.sin(2 * np.pi * (hour_of_day - 8) / 24)

    # Interaction: daily variation is larger in summer
    seasonal_interaction = 0.3 * daily_amplitude * np.sin(2 * np.pi * (day_of_year - 105) / 365) * \
                          np.sin(2 * np.pi * (hour_of_day - 8) / 24)

    # Random walk component for weather patterns (autocorrelated noise)
    weather_pattern = np.zeros(n_hours)
    weather_pattern[0] = np.random.normal(0, noise_std)
    for i in range(1, n_hours):
        weather_pattern[i] = 0.98 * weather_pattern[i-1] + np.random.normal(0, noise_std * 0.2)

    # White noise
    noise = np.random.normal(0, noise_std * 0.5, n_hours)

    # Combine all components
    temperature = base_temp + annual_seasonal + daily_seasonal + seasonal_interaction + weather_pattern + noise

    df = pd.DataFrame({
        'unique_id': 'temp_station_1',
        'ds': dates,
        'y': temperature
    })

    return df


def train_and_forecast(df, horizon=168, input_size=336):
    """
    Train NeuralForecast models and generate forecasts.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data with columns ['unique_id', 'ds', 'y']
    horizon : int
        Forecast horizon in hours (default: 168 = 1 week)
    input_size : int
        Number of historical observations to use (default: 336 = 2 weeks)

    Returns:
    --------
    tuple
        (forecasts DataFrame, fitted NeuralForecast object)
    """
    print(f"Training models with input_size={input_size}, horizon={horizon}")

    models = [
        NHITS(
            input_size=input_size,
            h=horizon,
            max_steps=200,
            n_pool_kernel_size=[2, 2, 2],
            n_freq_downsample=[24, 4, 1],  # Capture daily patterns
            scaler_type='standard',
            learning_rate=1e-3,
            batch_size=32,
            random_seed=42
        ),
        NBEATS(
            input_size=input_size,
            h=horizon,
            max_steps=200,
            scaler_type='standard',
            learning_rate=1e-3,
            batch_size=32,
            random_seed=42
        ),
    ]

    nf = NeuralForecast(models=models, freq='h')
    nf.fit(df=df)
    forecasts = nf.predict()

    return forecasts, nf


def evaluate_models(df, horizon=168, n_windows=3):
    """
    Perform cross-validation to evaluate model performance.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    horizon : int
        Forecast horizon in hours
    n_windows : int
        Number of cross-validation windows

    Returns:
    --------
    dict
        Dictionary with evaluation metrics for each model
    """
    print(f"\nRunning cross-validation with {n_windows} windows...")

    input_size = horizon * 2  # Use 2x horizon as input

    models = [
        NHITS(
            input_size=input_size,
            h=horizon,
            max_steps=100,
            scaler_type='standard',
            random_seed=42
        ),
        NBEATS(
            input_size=input_size,
            h=horizon,
            max_steps=100,
            scaler_type='standard',
            random_seed=42
        ),
    ]

    nf = NeuralForecast(models=models, freq='h')
    cv_results = nf.cross_validation(df=df, n_windows=n_windows)

    # Calculate metrics
    metrics = {}
    model_names = ['NHITS', 'NBEATS']

    for model_name in model_names:
        errors = cv_results['y'] - cv_results[model_name]
        metrics[model_name] = {
            'MAE': np.mean(np.abs(errors)),
            'RMSE': np.sqrt(np.mean(errors**2)),
            'MAPE': np.mean(np.abs(errors / cv_results['y'])) * 100
        }

    return metrics, cv_results


def plot_results(df, forecasts, save_path='hourly_temp_forecast.png'):
    """
    Plot historical data and forecasts.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Get last 2 weeks of historical data + forecast
    last_date = df['ds'].max()
    plot_start = last_date - pd.Timedelta(days=14)
    recent_data = df[df['ds'] >= plot_start]

    # Plot 1: Full view with forecast
    ax1 = axes[0]
    ax1.plot(recent_data['ds'], recent_data['y'],
             label='Historical', color='blue', linewidth=1, alpha=0.8)
    ax1.plot(forecasts['ds'], forecasts['NHITS'],
             label='NHITS Forecast', color='red', linewidth=2, linestyle='--')
    ax1.plot(forecasts['ds'], forecasts['NBEATS'],
             label='NBEATS Forecast', color='green', linewidth=2, linestyle=':')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Hourly Temperature: Historical Data and 1-Week Forecast')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Daily pattern visualization (last 3 days)
    ax2 = axes[1]
    last_3_days = df[df['ds'] >= (last_date - pd.Timedelta(days=3))]
    ax2.plot(last_3_days['ds'], last_3_days['y'],
             label='Actual', color='blue', linewidth=1.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Daily Temperature Pattern (Last 3 Days)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def print_data_summary(df):
    """
    Print summary statistics of the temperature data.
    """
    print("\n" + "=" * 60)
    print("HOURLY TEMPERATURE DATA SUMMARY")
    print("=" * 60)
    print(f"Time range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Total observations: {len(df):,}")
    print(f"Frequency: Hourly")
    print("\nTemperature Statistics:")
    print(f"  Mean:   {df['y'].mean():.2f}°C")
    print(f"  Std:    {df['y'].std():.2f}°C")
    print(f"  Min:    {df['y'].min():.2f}°C")
    print(f"  Max:    {df['y'].max():.2f}°C")
    print(f"  Range:  {df['y'].max() - df['y'].min():.2f}°C")

    # Daily statistics
    df_with_date = df.copy()
    df_with_date['date'] = df_with_date['ds'].dt.date
    daily_stats = df_with_date.groupby('date')['y'].agg(['min', 'max', 'mean'])
    print(f"\nDaily Temperature Range:")
    print(f"  Avg daily min:  {daily_stats['min'].mean():.2f}°C")
    print(f"  Avg daily max:  {daily_stats['max'].mean():.2f}°C")
    print(f"  Avg daily swing: {(daily_stats['max'] - daily_stats['min']).mean():.2f}°C")


def main():
    """
    Main function to run the hourly temperature forecasting demo.
    """
    print("=" * 60)
    print("HOURLY TEMPERATURE FORECASTING WITH NEURALFORECAST")
    print("=" * 60)

    # Step 1: Generate simulated hourly temperature data
    print("\n[Step 1] Generating simulated hourly temperature data...")
    df = generate_hourly_temperature_data(
        start_date='2022-01-01',
        n_days=365 * 2,  # 2 years of data
        base_temp=15.0,
        annual_amplitude=12.0,
        daily_amplitude=8.0,
        noise_std=2.0
    )
    print_data_summary(df)

    # Step 2: Train models and generate forecast
    print("\n[Step 2] Training neural network models...")
    horizon = 168  # 1 week ahead
    input_size = 336  # 2 weeks of history

    forecasts, nf = train_and_forecast(df, horizon=horizon, input_size=input_size)

    print("\n" + "=" * 60)
    print("FORECAST RESULTS (Next 7 Days)")
    print("=" * 60)
    print(forecasts.head(24))  # First 24 hours
    print(f"... ({len(forecasts)} total hourly forecasts)")

    # Step 3: Cross-validation
    print("\n[Step 3] Evaluating models with cross-validation...")
    metrics, cv_results = evaluate_models(df, horizon=72, n_windows=3)  # 3-day forecast for CV

    print("\n" + "=" * 60)
    print("MODEL EVALUATION METRICS")
    print("=" * 60)
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        print(f"  MAE:  {model_metrics['MAE']:.2f}°C")
        print(f"  RMSE: {model_metrics['RMSE']:.2f}°C")
        print(f"  MAPE: {model_metrics['MAPE']:.2f}%")

    # Step 4: Plot results
    print("\n[Step 4] Generating plots...")
    plot_results(df, forecasts)

    # Save forecast to CSV
    forecasts.to_csv('hourly_temperature_forecast.csv', index=False)
    print("Forecast saved to hourly_temperature_forecast.csv")

    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return df, forecasts, metrics


if __name__ == "__main__":
    df, forecasts, metrics = main()
