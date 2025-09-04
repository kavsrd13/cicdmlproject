import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("../data/adult_income.csv")

# 2. Features & target
X = df.drop("income", axis=1)
y = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

# 3. Categorical & numerical features
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
num_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# 4. Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

# 5. Pipeline with Logistic Regression
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=500))
    ]
)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train model
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 9. Save trained model
joblib.dump(model, "../model/model.pkl")
print("Model saved as model.pkl")
