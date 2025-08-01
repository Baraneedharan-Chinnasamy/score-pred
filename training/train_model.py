import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('cleanedtrain.csv')

cols_to_normalize = [
    'Days_since_launch',
    'Total_Stock_Sold_Percentage',
    'Alltime_Perday_Quantity',
    'Alltime_Perday_View',
    'Alltime_Perday_ATC',
    'Days_Since_Last_Sale'
]

scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df[cols_to_normalize]), columns=cols_to_normalize)
df_norm['Days_since_launch'] = 1 - df_norm['Days_since_launch']

# Copy Score to normalized df
df_norm['Score'] = df['Score']

features = cols_to_normalize
X = df_norm[features]
y = df_norm['Score']

best_rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    max_depth=None,
    random_state=42
)
best_rf.fit(X, y)

joblib.dump(best_rf, 'rf_selling_score_best.joblib')
joblib.dump(scaler, 'scaler.joblib')

