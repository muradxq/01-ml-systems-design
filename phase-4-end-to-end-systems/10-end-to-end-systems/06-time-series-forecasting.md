# Time Series Forecasting Systems

## Overview

Time series forecasting systems predict future values based on historical temporal data. These systems are critical for demand planning, financial forecasting, resource optimization, and anomaly detection. Time series presents unique challenges including seasonality, trends, missing data, and concept drift.

---

## 🎯 Problem Definition

### Common Use Cases

| Application | Forecast Horizon | Update Frequency |
|-------------|-----------------|------------------|
| **Demand Forecasting** | Days to months | Daily |
| **Financial Prediction** | Minutes to days | Real-time |
| **Resource Planning** | Hours to weeks | Hourly |
| **Anomaly Detection** | Real-time | Continuous |
| **Weather Forecasting** | Hours to days | Hourly |

### Requirements (Demand Forecasting Example)

| Requirement | Specification |
|-------------|---------------|
| **Latency** | < 1s per forecast |
| **Accuracy** | MAPE < 15% |
| **Horizon** | 1-90 days |
| **Granularity** | Daily, weekly |
| **Scale** | 100K+ time series |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Time Series Forecasting Architecture                       │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                    Data Ingestion Layer                         │         │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │         │
│  │  │  Streaming   │  │  Batch       │  │  External    │         │         │
│  │  │  (Kafka)     │  │  (S3/GCS)    │  │  (APIs)      │         │         │
│  │  └──────────────┘  └──────────────┘  └──────────────┘         │         │
│  └───────────────────────────────┬─────────────────────────────────┘        │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Feature Engineering Layer                      │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │  Lag         │  │  Rolling     │  │  Calendar    │           │        │
│  │  │  Features    │  │  Statistics  │  │  Features    │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │  Fourier     │  │  External    │  │  Embeddings  │           │        │
│  │  │  (Seasonal)  │  │  Regressors  │  │  (Entity)    │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  └───────────────────────────────┬─────────────────────────────────┘        │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Model Layer                                    │        │
│  │  ┌──────────────────────────────────────────────────────────┐   │        │
│  │  │  Ensemble of Models                                       │   │        │
│  │  │  - Statistical: ARIMA, ETS, Theta                        │   │        │
│  │  │  - ML: LightGBM, XGBoost                                 │   │        │
│  │  │  - Deep Learning: LSTM, Transformer, N-BEATS             │   │        │
│  │  └──────────────────────────────────────────────────────────┘   │        │
│  └───────────────────────────────┬─────────────────────────────────┘        │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    Serving Layer                                  │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │  Forecast    │──│  Uncertainty │──│  Post-       │─▶ Output  │        │
│  │  │  Generation  │  │  Quantiles   │  │  Processing  │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  └─────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Deep Dive

### 1. Time Series Feature Engineering

```python
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TimeSeriesFeatureEngine:
    """Feature engineering for time series forecasting."""
    
    def __init__(
        self,
        lag_windows: List[int] = [1, 7, 14, 28, 365],
        rolling_windows: List[int] = [7, 14, 28, 90],
        date_features: bool = True,
        fourier_order: int = 5
    ):
        self.lag_windows = lag_windows
        self.rolling_windows = rolling_windows
        self.date_features = date_features
        self.fourier_order = fourier_order
    
    def create_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str
    ) -> pd.DataFrame:
        """Create all features for time series."""
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Lag features
        for lag in self.lag_windows:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window).std()
            df[f'rolling_min_{window}'] = df[target_col].shift(1).rolling(window).min()
            df[f'rolling_max_{window}'] = df[target_col].shift(1).rolling(window).max()
        
        # Expanding statistics
        df['expanding_mean'] = df[target_col].shift(1).expanding().mean()
        
        # Date features
        if self.date_features:
            df = self._add_date_features(df, date_col)
        
        # Fourier features for seasonality
        df = self._add_fourier_features(df, date_col)
        
        return df
    
    def _add_date_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Add calendar-based features."""
        
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['year'] = df[date_col].dt.year
        
        # Boolean features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        
        return df
    
    def _add_fourier_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Add Fourier features for seasonality."""
        
        # Yearly seasonality
        day_of_year = df[date_col].dt.dayofyear
        for k in range(1, self.fourier_order + 1):
            df[f'sin_year_{k}'] = np.sin(2 * np.pi * k * day_of_year / 365.25)
            df[f'cos_year_{k}'] = np.cos(2 * np.pi * k * day_of_year / 365.25)
        
        # Weekly seasonality
        day_of_week = df[date_col].dt.dayofweek
        for k in range(1, min(self.fourier_order, 3) + 1):
            df[f'sin_week_{k}'] = np.sin(2 * np.pi * k * day_of_week / 7)
            df[f'cos_week_{k}'] = np.cos(2 * np.pi * k * day_of_week / 7)
        
        return df
    
    def create_future_features(
        self,
        last_date: datetime,
        horizon: int,
        history_df: pd.DataFrame,
        target_col: str,
        date_col: str
    ) -> pd.DataFrame:
        """Create features for future dates (inference time)."""
        
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        future_df = pd.DataFrame({date_col: future_dates})
        
        # Date features
        if self.date_features:
            future_df = self._add_date_features(future_df, date_col)
        
        # Fourier features
        future_df = self._add_fourier_features(future_df, date_col)
        
        # For lag features, we need to iteratively predict
        # This is handled in the forecasting loop
        
        return future_df

class ExternalFeatures:
    """External/exogenous features for forecasting."""
    
    def __init__(self, feature_sources: Dict):
        self.sources = feature_sources
    
    def get_features(
        self,
        entity_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get external features for entity and date range."""
        
        features = {}
        
        # Weather features
        if 'weather' in self.sources:
            weather = self.sources['weather'].get(
                start_date=start_date,
                end_date=end_date
            )
            features.update(weather)
        
        # Holiday features
        if 'holidays' in self.sources:
            holidays = self.sources['holidays'].get(
                start_date=start_date,
                end_date=end_date
            )
            features['is_holiday'] = holidays
        
        # Economic indicators
        if 'economic' in self.sources:
            economic = self.sources['economic'].get(
                start_date=start_date,
                end_date=end_date
            )
            features.update(economic)
        
        # Promotions/events
        if 'promotions' in self.sources:
            promos = self.sources['promotions'].get(
                entity_id=entity_id,
                start_date=start_date,
                end_date=end_date
            )
            features['has_promotion'] = promos
        
        return pd.DataFrame(features)
```

### 2. Forecasting Models

```python
from abc import ABC, abstractmethod
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

class Forecaster(ABC):
    """Base forecaster interface."""
    
    @abstractmethod
    def fit(self, y: np.ndarray, X: np.ndarray = None):
        pass
    
    @abstractmethod
    def predict(self, horizon: int, X: np.ndarray = None) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_interval(
        self,
        horizon: int,
        X: np.ndarray = None,
        alpha: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return point forecast and prediction intervals."""
        pass

class StatisticalForecaster(Forecaster):
    """Statistical forecasting models (ARIMA, ETS, etc.)."""
    
    def __init__(self, model_type: str = "auto_arima"):
        self.model_type = model_type
        self.model = None
    
    def fit(self, y: np.ndarray, X: np.ndarray = None):
        """Fit statistical model."""
        from pmdarima import auto_arima
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        if self.model_type == "auto_arima":
            self.model = auto_arima(
                y,
                X=X,
                seasonal=True,
                m=7,  # Weekly seasonality
                suppress_warnings=True,
                stepwise=True
            )
        elif self.model_type == "ets":
            self.model = ExponentialSmoothing(
                y,
                trend='add',
                seasonal='add',
                seasonal_periods=7
            ).fit()
    
    def predict(self, horizon: int, X: np.ndarray = None) -> np.ndarray:
        """Generate point forecasts."""
        if self.model_type == "auto_arima":
            return self.model.predict(n_periods=horizon, X=X)
        else:
            return self.model.forecast(horizon)
    
    def predict_interval(
        self,
        horizon: int,
        X: np.ndarray = None,
        alpha: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecast with prediction intervals."""
        
        if self.model_type == "auto_arima":
            forecast, conf_int = self.model.predict(
                n_periods=horizon,
                X=X,
                return_conf_int=True,
                alpha=alpha
            )
            return forecast, conf_int[:, 0], conf_int[:, 1]
        else:
            # Simulate for ETS
            forecast = self.model.forecast(horizon)
            std = np.std(self.model.resid) * np.sqrt(np.arange(1, horizon + 1))
            z = 1.96
            return forecast, forecast - z * std, forecast + z * std

class MLForecaster(Forecaster):
    """Machine learning forecasters (LightGBM, XGBoost)."""
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        quantiles: List[float] = [0.05, 0.5, 0.95]
    ):
        self.model_type = model_type
        self.quantiles = quantiles
        self.models = {}
    
    def fit(self, y: np.ndarray, X: np.ndarray):
        """Fit ML models for each quantile."""
        
        for q in self.quantiles:
            if self.model_type == "lightgbm":
                model = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=q,
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=31
                )
            else:
                model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=q,
                    n_estimators=100
                )
            
            model.fit(X, y)
            self.models[q] = model
    
    def predict(self, horizon: int, X: np.ndarray) -> np.ndarray:
        """Generate point forecasts (median)."""
        return self.models[0.5].predict(X[:horizon])
    
    def predict_interval(
        self,
        horizon: int,
        X: np.ndarray,
        alpha: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecast with prediction intervals."""
        
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        
        point = self.models[0.5].predict(X[:horizon])
        lower = self.models.get(lower_q, self.models[min(self.quantiles)]).predict(X[:horizon])
        upper = self.models.get(upper_q, self.models[max(self.quantiles)]).predict(X[:horizon])
        
        return point, lower, upper

class DeepForecaster(Forecaster):
    """Deep learning forecasters (LSTM, Transformer)."""
    
    def __init__(
        self,
        model_type: str = "lstm",
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        lookback: int = 30
    ):
        import torch
        import torch.nn as nn
        
        self.model_type = model_type
        self.lookback = lookback
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == "lstm":
            self.model = self._create_lstm(input_size, hidden_size, num_layers)
        else:
            self.model = self._create_transformer(input_size, hidden_size)
        
        self.model.to(self.device)
    
    def _create_lstm(self, input_size, hidden_size, num_layers):
        import torch.nn as nn
        
        class LSTMForecaster(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=0.1
                )
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out
        
        return LSTMForecaster(input_size, hidden_size, num_layers)
    
    def fit(self, y: np.ndarray, X: np.ndarray = None, epochs: int = 100):
        """Train deep learning model."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # Prepare sequences
        sequences, targets = self._create_sequences(y, self.lookback)
        
        dataset = TensorDataset(
            torch.FloatTensor(sequences),
            torch.FloatTensor(targets)
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
    
    def predict(self, horizon: int, X: np.ndarray = None) -> np.ndarray:
        """Generate multi-step forecast."""
        import torch
        
        self.model.eval()
        predictions = []
        
        # Start with last lookback values
        current_input = torch.FloatTensor(X[-self.lookback:]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(horizon):
                pred = self.model(current_input)
                predictions.append(pred.item())
                
                # Update input
                new_input = torch.cat([
                    current_input[:, 1:, :],
                    pred.unsqueeze(1).unsqueeze(2)
                ], dim=1)
                current_input = new_input
        
        return np.array(predictions)
    
    def _create_sequences(self, data, lookback):
        """Create sequences for training."""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X).reshape(-1, lookback, 1), np.array(y)

class EnsembleForecaster:
    """Ensemble of multiple forecasters."""
    
    def __init__(
        self,
        forecasters: Dict[str, Forecaster],
        weights: Dict[str, float] = None
    ):
        self.forecasters = forecasters
        self.weights = weights or {k: 1/len(forecasters) for k in forecasters}
    
    def fit(self, y: np.ndarray, X: np.ndarray = None):
        """Fit all forecasters."""
        for name, forecaster in self.forecasters.items():
            forecaster.fit(y, X)
    
    def predict(self, horizon: int, X: np.ndarray = None) -> np.ndarray:
        """Generate ensemble forecast."""
        forecasts = {}
        
        for name, forecaster in self.forecasters.items():
            forecasts[name] = forecaster.predict(horizon, X)
        
        # Weighted average
        ensemble = sum(
            self.weights[name] * forecast
            for name, forecast in forecasts.items()
        )
        
        return ensemble
    
    def optimize_weights(
        self,
        y_val: np.ndarray,
        X_val: np.ndarray = None
    ):
        """Optimize ensemble weights based on validation performance."""
        from scipy.optimize import minimize
        
        forecasts = {}
        for name, forecaster in self.forecasters.items():
            forecasts[name] = forecaster.predict(len(y_val), X_val)
        
        def objective(weights):
            weights_dict = dict(zip(self.forecasters.keys(), weights))
            ensemble = sum(
                weights_dict[name] * forecast
                for name, forecast in forecasts.items()
            )
            return np.mean((ensemble - y_val) ** 2)
        
        # Optimize
        n = len(self.forecasters)
        result = minimize(
            objective,
            x0=[1/n] * n,
            constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1},
            bounds=[(0, 1)] * n
        )
        
        self.weights = dict(zip(self.forecasters.keys(), result.x))
```

### 3. Complete Forecasting Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI()

class ForecastRequest(BaseModel):
    entity_id: str
    horizon: int = 30
    confidence_level: float = 0.9
    include_history: bool = False

class ForecastPoint(BaseModel):
    date: str
    forecast: float
    lower_bound: float
    upper_bound: float

class ForecastResponse(BaseModel):
    entity_id: str
    forecasts: List[ForecastPoint]
    model_used: str
    metrics: Dict[str, float]

class ForecastingService:
    """Complete forecasting service."""
    
    def __init__(
        self,
        feature_engine: TimeSeriesFeatureEngine,
        model_registry: Dict[str, EnsembleForecaster],
        data_store,
        model_selector
    ):
        self.features = feature_engine
        self.models = model_registry
        self.data_store = data_store
        self.selector = model_selector
    
    async def forecast(self, request: ForecastRequest) -> ForecastResponse:
        """Generate forecast for entity."""
        
        # Get historical data
        history = await self.data_store.get_history(
            entity_id=request.entity_id,
            lookback_days=365 * 2
        )
        
        if len(history) < 30:
            raise HTTPException(
                status_code=400,
                detail="Insufficient history for forecasting"
            )
        
        # Select best model for this entity
        model_key = self.selector.select_model(history)
        model = self.models[model_key]
        
        # Create features
        df = self.features.create_features(
            history,
            target_col='value',
            date_col='date'
        )
        
        # Fit model
        y = df['value'].dropna().values
        X = df.drop(['value', 'date'], axis=1).dropna().values
        model.fit(y[:-30], X[:-30])  # Hold out last 30 for validation
        
        # Generate future features
        last_date = history['date'].max()
        future_df = self.features.create_future_features(
            last_date=last_date,
            horizon=request.horizon,
            history_df=history,
            target_col='value',
            date_col='date'
        )
        
        # Generate forecast with intervals
        alpha = 1 - request.confidence_level
        point, lower, upper = model.predict_interval(
            horizon=request.horizon,
            X=future_df.drop('date', axis=1).values,
            alpha=alpha
        )
        
        # Format response
        forecasts = []
        for i, date in enumerate(future_df['date']):
            forecasts.append(ForecastPoint(
                date=date.strftime('%Y-%m-%d'),
                forecast=float(point[i]),
                lower_bound=float(lower[i]),
                upper_bound=float(upper[i])
            ))
        
        # Calculate validation metrics
        metrics = self._calculate_metrics(y[-30:], model.predict(30, X[-30:]))
        
        return ForecastResponse(
            entity_id=request.entity_id,
            forecasts=forecasts,
            model_used=model_key,
            metrics=metrics
        )
    
    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate forecast accuracy metrics."""
        
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        return {
            "mae": float(mae),
            "mape": float(mape),
            "rmse": float(rmse)
        }

class ModelSelector:
    """Select best model for each time series."""
    
    def __init__(self, performance_store):
        self.performance = performance_store
    
    def select_model(self, history: pd.DataFrame) -> str:
        """Select model based on series characteristics."""
        
        # Analyze series
        characteristics = self._analyze_series(history)
        
        # Rule-based selection
        if characteristics['is_sparse']:
            return 'croston'  # For intermittent demand
        elif characteristics['has_strong_trend']:
            return 'prophet'  # Good with trends
        elif characteristics['is_highly_seasonal']:
            return 'ets'  # Good with seasonality
        else:
            return 'ensemble'  # Default ensemble
    
    def _analyze_series(self, history: pd.DataFrame) -> Dict:
        """Analyze time series characteristics."""
        
        values = history['value'].values
        
        return {
            'is_sparse': np.mean(values == 0) > 0.3,
            'has_strong_trend': self._has_trend(values),
            'is_highly_seasonal': self._is_seasonal(values),
            'length': len(values),
            'volatility': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        }
    
    def _has_trend(self, values: np.ndarray) -> bool:
        """Check for significant trend."""
        from scipy.stats import linregress
        
        x = np.arange(len(values))
        slope, _, r_value, p_value, _ = linregress(x, values)
        
        return p_value < 0.05 and abs(r_value) > 0.5
    
    def _is_seasonal(self, values: np.ndarray) -> bool:
        """Check for significant seasonality."""
        from scipy.signal import periodogram
        
        if len(values) < 14:
            return False
        
        f, Pxx = periodogram(values)
        
        # Check for peaks at weekly frequency
        weekly_idx = np.argmin(np.abs(f - 1/7))
        
        return Pxx[weekly_idx] > np.median(Pxx) * 5

# API Endpoint
forecasting_service = ForecastingService(...)

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    return await forecasting_service.forecast(request)
```

---

## 📈 Metrics & Evaluation

### Forecast Accuracy Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **MAE** | Mean Absolute Error | General |
| **MAPE** | Mean Absolute % Error | Relative comparison |
| **RMSE** | Root Mean Square Error | Penalize large errors |
| **MASE** | Mean Absolute Scaled Error | Compare across series |
| **Coverage** | % in prediction interval | Uncertainty quality |

### System Metrics

```python
from prometheus_client import Histogram, Gauge, Counter

forecast_latency = Histogram(
    'forecast_latency_seconds',
    'Forecast generation latency',
    ['model']
)

forecast_accuracy = Gauge(
    'forecast_mape',
    'Rolling MAPE by entity',
    ['entity_id']
)

forecasts_generated = Counter(
    'forecasts_generated_total',
    'Total forecasts generated',
    ['model']
)
```

---

## ⚖️ Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Model Complexity** | Simple (fast, interpretable) | Complex (accurate) |
| **Update Frequency** | Frequent (fresh) | Infrequent (stable) |
| **Granularity** | Fine (detailed) | Coarse (robust) |
| **Global vs Local** | Global model (scalable) | Local per series (accurate) |

---

## 🎤 Interview Tips

**Common Questions:**
1. How do you handle missing values?
2. How do you detect and handle outliers?
3. How do you choose between models?
4. How do you handle seasonality?
5. How do you scale to millions of time series?

**Key Points:**
- Feature engineering is critical
- Ensemble often beats single models
- Prediction intervals for uncertainty
- Backtesting for evaluation
- Automated model selection

---

## 🔗 Related Topics

- [Feature Engineering](../../phase-2-core-components/03-feature-engineering/00-README.md)
- [Data Quality](../../phase-2-core-components/02-data-management/04-data-quality.md)
- [Model Monitoring](../../phase-3-operations-and-reliability/06-monitoring-observability/01-model-monitoring.md)
