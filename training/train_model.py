import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('traindata1.csv')

cols_to_normalize = [
    'Days_since_launch',
    'Total_Stock_Sold_Percentage',
    'Alltime_Perday_Quantity',
    'Alltime_Perday_View',
    'Alltime_Perday_ATC'
]

scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df[cols_to_normalize]), columns=cols_to_normalize)
df_norm['Days_since_launch'] = 1 - df_norm['Days_since_launch']

score = (
    0.10 * df_norm['Days_since_launch'] +
    0.30 * df_norm['Total_Stock_Sold_Percentage'] +
    0.25 * df_norm['Alltime_Perday_Quantity'] +
    0.15 * df_norm['Alltime_Perday_View'] +
    0.20 * df_norm['Alltime_Perday_ATC']
)

df['Score'] = pd.qcut(score, 5, labels=[1,2,3,4,5]).astype(int)
df['Score'] = 6 - df['Score']

features = cols_to_normalize
X = df[features]
y = df['Score']

best_rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    max_depth=None,
    random_state=42
)
best_rf.fit(X, y)

# Save model to training folder
joblib.dump(best_rf, 'rf_selling_score_best.joblib')
