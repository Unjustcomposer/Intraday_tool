import pandas as pd
import numpy as np

def time_of_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode intraday time-of-day features — critical for intraday alpha.
    
    NSE session structure:
      09:15 - 09:30  Opening range (high volatility, momentum)
      09:30 - 11:30  Morning session (trend establishment)
      11:30 - 13:30  Lunch lull (low volume, mean reversion)
      13:30 - 14:30  Afternoon session (trend continuation)
      14:30 - 15:15  Closing session (high volatility, institutional flow)
      15:15 - 15:30  Last 15 min (position squaring, high urgency)
    
    Returns DataFrame with columns:
      - time_minutes: minutes since market open (0-375)
      - time_sin, time_cos: cyclical encoding
      - session_opening, session_morning, session_lunch, session_afternoon, session_closing: one-hot
      - minutes_to_close: minutes remaining until 15:30
    """
    features = pd.DataFrame(index=df.index)
    
    # Extract time information
    if hasattr(df.index, 'hour'):
        hours = df.index.hour
        minutes = df.index.minute
    elif 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        hours = ts.dt.hour
        minutes = ts.dt.minute
    else:
        # No time information available — return zeros
        for col in ['time_minutes', 'time_sin', 'time_cos', 'session_opening',
                     'session_morning', 'session_lunch', 'session_afternoon',
                     'session_closing', 'minutes_to_close']:
            features[col] = 0.0
        return features
    
    # Minutes since market open (9:15 = 0)
    time_minutes = (hours - 9) * 60 + minutes - 15
    time_minutes = np.clip(time_minutes, 0, 375)
    features['time_minutes'] = time_minutes
    
    # Cyclical encoding (avoids discontinuity at day boundaries)
    total_session_minutes = 375  # 9:15 to 15:30
    features['time_sin'] = np.sin(2 * np.pi * time_minutes / total_session_minutes)
    features['time_cos'] = np.cos(2 * np.pi * time_minutes / total_session_minutes)
    
    # Session one-hot encoding
    hour_min = hours * 100 + minutes
    features['session_opening'] = ((hour_min >= 915) & (hour_min < 930)).astype(float)
    features['session_morning'] = ((hour_min >= 930) & (hour_min < 1130)).astype(float)
    features['session_lunch'] = ((hour_min >= 1130) & (hour_min < 1330)).astype(float)
    features['session_afternoon'] = ((hour_min >= 1330) & (hour_min < 1430)).astype(float)
    features['session_closing'] = ((hour_min >= 1430) & (hour_min <= 1530)).astype(float)
    
    # Minutes to close
    close_minutes = 15 * 60 + 30  # 15:30 in minutes
    current_minutes = hours * 60 + minutes
    features['minutes_to_close'] = np.clip(close_minutes - current_minutes, 0, 375)
    
    return features
