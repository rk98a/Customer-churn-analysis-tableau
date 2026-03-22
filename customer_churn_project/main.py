import pandas as pd


df = pd.read_csv("data/telco.csv")
df.columns = df.columns.str.strip()


df = df[df["Customer Status"] != "Joined"]

df["Churn"] = df["Customer Status"].map({
    "Churned": 1,
    "Stayed": 0
})

df["Churn"] = df["Churn"].astype(int)


drop_cols = [
    "Customer ID",
    "Customer Status",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Category",
    "Churn Reason"
]

df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)


for col in df.select_dtypes(include=['int64','float64']).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])


X = df.drop("Churn", axis=1)
y = df["Churn"]


X = pd.get_dummies(X, drop_first=True)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


from sklearn.metrics import roc_auc_score

y_prob = rf_model.predict_proba(X_test)[:,1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight={0:1, 1:2},  # focus more on churn
    random_state=42
)
df.to_csv("cleaned_churn_data.csv", index=False)