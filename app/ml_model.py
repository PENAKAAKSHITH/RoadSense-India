# ============================================================
# ROADSENSE INDIA — ML Model (Linear Regression)
# File: app/ml_model.py
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go


def load_data(path: str = '../data/cleaned.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def train_model(df: pd.DataFrame):
    """
    Train a Linear Regression model to predict accident count.
    Returns: model, feature_cols, r2, rmse, fig
    """
    data = df.copy().dropna(subset=['accidents', 'killed', 'injured'])

    le_state = LabelEncoder()
    data['state_encoded'] = le_state.fit_transform(data['state'].astype(str))

    feature_cols = ['state_encoded', 'year', 'killed', 'injured']

    if 'road_type' in data.columns:
        le_road = LabelEncoder()
        data['road_type_encoded'] = le_road.fit_transform(data['road_type'].astype(str))
        feature_cols.append('road_type_encoded')

    X = data[feature_cols]
    y = data['accidents']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = round(r2_score(y_test, y_pred), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 1)

    # ── Actual vs Predicted scatter ──────────────────────────
    result_df = pd.DataFrame({
        'Actual'   : y_test.values,
        'Predicted': y_pred
    })

    fig = px.scatter(
        result_df,
        x='Actual',
        y='Predicted',
        opacity=0.55,
        color_discrete_sequence=['#E8593C'],
        title=f'Linear Regression — Actual vs Predicted  |  R² = {r2}  |  RMSE = {rmse:,.0f}',
        labels={'Actual': 'Actual Accidents', 'Predicted': 'Predicted Accidents'}
    )
    # Perfect-prediction reference line
    lo = result_df['Actual'].min()
    hi = result_df['Actual'].max()
    fig.add_shape(
        type='line', x0=lo, y0=lo, x1=hi, y1=hi,
        line=dict(color='#3B8BD4', dash='dash', width=1.5)
    )
    fig.update_layout(height=440)

    # ── Feature importance (coefficients) bar ────────────────
    coef_df = pd.DataFrame({
        'Feature'    : feature_cols,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=True)

    coef_fig = px.bar(
        coef_df,
        x='Coefficient',
        y='Feature',
        orientation='h',
        color='Coefficient',
        color_continuous_scale='RdBu',
        title='Feature Coefficients (importance)',
    )
    coef_fig.update_layout(height=300)

    print(f"✅ Model trained  |  R² = {r2}  |  RMSE = {rmse:,.0f}")
    print(f"   Features used  : {feature_cols}")

    return model, feature_cols, r2, rmse, fig, coef_fig


def predict_single(model, feature_cols: list,
                   state_encoded: int, year: int,
                   killed: int, injured: int,
                   road_type_encoded: int = 0) -> float:
    """Quick single prediction helper used by the dashboard."""
    row = {'state_encoded': state_encoded, 'year': year,
           'killed': killed, 'injured': injured}
    if 'road_type_encoded' in feature_cols:
        row['road_type_encoded'] = road_type_encoded
    X = pd.DataFrame([row])[feature_cols]
    return max(0, round(model.predict(X)[0], 0))


# ── Stand-alone test ─────────────────────────────────────────
if __name__ == '__main__':
    df = load_data()
    model, feature_cols, r2, rmse, fig, coef_fig = train_model(df)
    fig.show()
    coef_fig.show()