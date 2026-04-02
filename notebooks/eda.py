# ============================================================
# ROADSENSE INDIA — Full EDA Script (FINAL FIXED)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# LOAD DATA (safe path)
# ------------------------------------------------------------
df = pd.read_csv('C:\\Users\\penak\\OneDrive\\Desktop\\PROJECT\\RoadSense-India\\data\\accidents_raw.csv')

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------------------------------------------------
# BASIC INFO
# ------------------------------------------------------------
print(df.dtypes)
print(df.isnull().sum())

# ------------------------------------------------------------
# CLEANING
# ------------------------------------------------------------

# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
)

# Rename columns (based on YOUR dataset)
df.rename(columns={
    'state_name': 'state',
    'number_of_casualties': 'accidents',
    'number_of_fatalities': 'killed'
}, inplace=True)

# Create injured column
df['injured'] = df['accidents'] - df['killed']

# Convert numeric
for col in ['accidents', 'killed', 'injured', 'year']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop invalid rows
df.dropna(subset=['state', 'accidents'], inplace=True)

# ------------------------------------------------------------
# FEATURE ENGINEERING (FIXED — NO INF)
# ------------------------------------------------------------

df['fatality_rate'] = (
    df['killed'] / df['accidents'].replace(0, np.nan)
) * 100

df['fatality_rate'] = df['fatality_rate'].fillna(0)

df['severity_score'] = (
    (df['killed'] * 2 + df['injured']) /
    df['accidents'].replace(0, np.nan)
)

df['severity_score'] = df['severity_score'].fillna(0)

# Save cleaned data
df.to_csv('C:\\Users\\penak\\OneDrive\\Desktop\\PROJECT\\RoadSense-India\\data\\cleaned.csv', index=False)

print("✅ Cleaning done")

# ------------------------------------------------------------
# AGGREGATION (FIXED LOGIC)
# ------------------------------------------------------------

state_agg = (
    df.groupby('state')
    .agg(
        total_accidents=('accidents', 'sum'),
        total_killed=('killed', 'sum'),
        total_injured=('injured', 'sum')
    )
    .reset_index()
)

# ✅ Correct fatality rate calculation (after aggregation)
state_agg['avg_fatality_rate'] = (
    state_agg['total_killed'] /
    state_agg['total_accidents']
) * 100

state_agg = state_agg.sort_values('total_accidents', ascending=False)

top15 = state_agg.head(15)

print("\nTop 5 states:\n", top15.head())

# ------------------------------------------------------------
# VISUALIZATION (NON-BLOCKING)
# ------------------------------------------------------------

# 1. Top states bar chart (saved instead of shown)
fig1 = px.bar(
    top15,
    x='total_accidents',
    y='state',
    orientation='h',
    color='avg_fatality_rate',
    title='Top 15 States by Accidents'
)

fig1.write_html("top_states.html")

# 2. Year trend
yearly = df.groupby('year').agg(
    accidents=('accidents','sum'),
    killed=('killed','sum')
).reset_index()

fig2 = px.line(
    yearly,
    x='year',
    y='accidents',
    title='Yearly Accident Trend'
)

fig2.write_html("year_trend.html")

# 3. Heatmap (saved as image)
plt.figure()
corr = df[['accidents','killed','injured','fatality_rate']].corr()
sns.heatmap(corr, annot=True)

plt.savefig("correlation_heatmap.png")
plt.close()

# ------------------------------------------------------------
# DONE
# ------------------------------------------------------------
print("\n✅ EDA COMPLETED SUCCESSFULLY")
print("📁 Files generated:")
print(" - data/cleaned.csv")
print(" - top_states.html")
print(" - year_trend.html")
print(" - correlation_heatmap.png")