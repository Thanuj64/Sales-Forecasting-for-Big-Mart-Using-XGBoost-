import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stButton>button {
    width: 100%;
    height: 50px;
    font-size:18px;
    border-radius:10px;
    background-color:#FF4B4B;
    color:white;
}

.result-box {
    background-color:#1c1f26;
    padding:25px;
    border-radius:12px;
    text-align:center;
    box-shadow:0px 0px 15px rgba(0,0,0,0.4);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("model/xgboost_regression.pkl")

model = load_model()
model_features = model.get_booster().feature_names

# ---------------- SALES CATEGORY ----------------
def sales_category(value):
    if value < 2000:
        return "Low"
    elif value < 4000:
        return "Medium"
    else:
        return "High"

# ---------------- HEADER ----------------
st.title("🛒 BigMart Sales Prediction Dashboard")
st.caption("AI Powered Retail Sales Forecasting System")
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")

input_method = st.sidebar.radio(
    "Prediction Mode",
    ["Manual Entry", "Upload CSV"]
)

output_type = st.sidebar.selectbox(
    "Output",
    ["Display Result", "Download CSV"]
)

# =====================================================
# ================= MANUAL ENTRY ======================
# =====================================================

if input_method == "Manual Entry":

    st.subheader("📦 Product Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        item_weight = st.number_input("Item Weight")
        item_visibility = st.number_input("Item Visibility")
        item_mrp = st.number_input("Item MRP")

    with col2:
        outlet_years = st.number_input("Outlet Years")

        item_fat_content = st.selectbox(
            "Item Fat Content",
            ["Low Fat", "Regular"]
        )

        item_type_combined = st.selectbox(
            "Item Type",
            ["Food", "Drinks", "Non-Consumable"]
        )

    with col3:
        outlet_size = st.selectbox(
            "Outlet Size",
            ["Small", "Medium", "High"]
        )

        outlet_location = st.selectbox(
            "Outlet Location",
            ["Tier 1", "Tier 2", "Tier 3"]
        )

        outlet_type = st.selectbox(
            "Outlet Type",
            ["Grocery Store",
             "Supermarket Type1",
             "Supermarket Type2",
             "Supermarket Type3"]
        )

    st.markdown("###")

    if st.button("🔮 Predict Sales"):

        if (
            item_weight == 0 or
            item_visibility == 0 or
            item_mrp == 0 or
            outlet_years == 0
        ):
            st.warning("⚠️ Please fill all values before prediction.")
            st.stop()

        data = {
            "Item_Weight": item_weight,
            "Item_Visibility": item_visibility,
            "Item_MRP": item_mrp,
            "Outlet_Years": outlet_years,

            "Item_Fat_Content_1": 1 if item_fat_content=="Low Fat" else 0,
            "Item_Fat_Content_2": 1 if item_fat_content=="Regular" else 0,

            "Outlet_Location_Type_1": 1 if outlet_location=="Tier 1" else 0,
            "Outlet_Location_Type_2": 1 if outlet_location=="Tier 2" else 0,

            "Outlet_Size_1": 1 if outlet_size=="Small" else 0,
            "Outlet_Size_2": 1 if outlet_size=="Medium" else 0,

            "Outlet_Type_1": 1 if outlet_type=="Grocery Store" else 0,
            "Outlet_Type_2": 1 if outlet_type=="Supermarket Type1" else 0,
            "Outlet_Type_3": 1 if outlet_type=="Supermarket Type2" else 0,

            "Item_Type_Combined_1": 1 if item_type_combined=="Food" else 0,
            "Item_Type_Combined_2": 1 if item_type_combined=="Drinks" else 0,
        }

        for i in range(1,10):
            data[f"Outlet_{i}"] = 0

        input_df = pd.DataFrame([data])

        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_features]

        prediction = model.predict(input_df)[0]

        st.markdown(
            f"""
            <div class="result-box">
                <h2>💰 Predicted Sales</h2>
                <h1>₹ {round(prediction,2)}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# ================= CSV UPLOAD ========================
# =====================================================

else:

    st.subheader("📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.success("✅ File Uploaded Successfully")
        st.dataframe(df.head())

        if st.button("🚀 Predict Batch Sales"):

            object_cols = df.select_dtypes(include=["object"]).columns
            df = df.drop(columns=object_cols, errors="ignore")

            for col in model_features:
                if col not in df.columns:
                    df[col] = 0

            df = df[model_features]

            predictions = model.predict(df)

            result_df = df.copy()
            result_df["Predicted_Sales"] = predictions
            result_df["Sales_Category"] = result_df[
                "Predicted_Sales"
            ].apply(sales_category)

            st.success("✅ Prediction Completed")

            if output_type == "Display Result":
                st.dataframe(result_df)

            else:
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download Predictions",
                    csv,
                    "bigmart_predictions.csv",
                    "text/csv"
                )