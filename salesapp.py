import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# =============================
# 📥 Load & Preprocess Data
# =============================
df = pd.read_csv("data.csv")
df.sale_date = pd.to_datetime(df.sale_date)

df = df[['quantity', 'total_price', 'payment_method', 'day_of_week', 'price', 'gender', 'age']]
df = df.rename(columns={'total_price': 'sales', 'payment_method': 'payment', 'day_of_week': 'day'})

def split_data(data):
    num_cols = data.select_dtypes(include=['number']).columns
    cat_cols = data.select_dtypes(include=['object']).columns
    return num_cols, cat_cols

x = df.drop('sales', axis=1)
y = df.sales
num_cols, cat_cols = split_data(x)

# =============================
# 🧪 Preprocessing Pipeline
# =============================
num_pipe = Pipeline([('scaler', StandardScaler())])
cat_pipe = Pipeline([('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

transformer = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
], remainder='passthrough')

# =============================
# 🎯 Train/Test Split
# =============================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# =============================
# 🤖 Model Pipelines
# =============================
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'KNN': KNeighborsRegressor(),
    'SVR': SVR(),
    'XGBoost': XGBRegressor()
}

model_pipes = {
    name: Pipeline([
        ('preprocessor', transformer),
        ('model', model)
    ])
    for name, model in models.items()
}

for pipe in model_pipes.values():
    pipe.fit(x_train, y_train)

# =============================
# 🌟 Streamlit UI
# =============================
st.set_page_config(page_title="🛍️ Sales Prediction App", layout="wide")
st.title("🛍️ Sales Prediction Web App")
st.markdown("Use this app to predict customer purchase amounts based on behavior and demographic data.")

st.markdown("### 📝 Enter Customer Details")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        quantity = st.number_input("🧺 Quantity", min_value=1, value=1)
        payment = st.selectbox("💳 Payment Method", sorted(df['payment'].unique()))
    with col2:
        price = st.number_input("💰 Price per Item", min_value=0.0, value=10.0)
        day = st.selectbox("📅 Day of Week", ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
    with col3:
        gender = st.selectbox("👤 Gender", sorted(df['gender'].unique()))
        age = st.number_input("🎂 Age", min_value=10, max_value=100, value=30)

    submit = st.form_submit_button("🔍 Predict Sales")

input_data = pd.DataFrame([{
    'quantity': quantity,
    'price': price,
    'payment': payment,
    'day': day,
    'gender': gender,
    'age': age
}])

# =============================
# 📊 Model Predictions
# =============================
if submit:
    st.markdown("---")
    st.subheader("📈 Predicted Sales by Model")
    tabs = st.tabs(list(model_pipes.keys()))

    for i, (name, pipe) in enumerate(model_pipes.items()):
        with tabs[i]:
            prediction = pipe.predict(input_data)[0]
            st.metric(label="🛒 Predicted Sales", value=f"N{prediction:,.2f}")

            # Evaluate on test data
            test_pred = pipe.predict(x_test)
            r2 = r2_score(y_test, test_pred)
            mae = mean_absolute_error(y_test, test_pred)
            mape = mean_absolute_percentage_error(y_test, test_pred)

            st.markdown("#### 🔍 Model Evaluation Metrics")
            st.write(f"• **R² Score**: `{r2:.3f}`")
            st.write(f"• **Mean Absolute Error (MAE)**: `N{mae:,.2f}`")
            st.write(f"• **Mean Absolute Percentage Error (MAPE)**: `{mape:.2%}`")

# =============================
# 🔻 Footer
# =============================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:gray; font-size: 13px;">
        🚀 Built with <a href="https://streamlit.io" target="_blank" style="color:#1f77b4;">Streamlit</a> |
        Created by <strong>Tolulope Emuleomo</strong> aka <strong>Data Professor</strong> 🧠<br>
        Connect: 
        <a href="https://twitter.com/dataprofessor_" target="_blank" style="color:#1DA1F2;">Twitter</a> | 
        <a href="https://github.com/dataprofessor290" target="_blank" style="color:#6e5494;">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
