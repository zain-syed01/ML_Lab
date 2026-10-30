from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).with_name("ai4i2020.csv")
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES = ["Type", *NUMERIC_FEATURES]
TARGET = "Machine failure"


@st.cache_resource
def load_model_and_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    features = data[FEATURES]
    target = data[TARGET]

    preprocessing = ColumnTransformer(
        [("product_type", OneHotEncoder(handle_unknown="ignore"), ["Type"])],
        remainder="passthrough",
    )
    model = make_pipeline(
        preprocessing,
        RandomForestClassifier(n_estimators=300, random_state=42),
    )
    model.fit(features, target)
    return model, data


st.set_page_config(page_title="Machine Failure Demo", page_icon="⚙️")
st.title("Machine Failure Classification")
st.write(
    "Try machine operating values and see how a Random Forest classifies "
    "the datapoint."
)
st.caption(
    "Educational demo using the AI4I 2020 dataset. "
    
)

try:
    model, data = load_model_and_data()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.subheader("About this model")
    st.write("Random Forest trained on the AI4I 2020 dataset.")
    st.write(f"Training rows: {len(data):,}")
    st.write(f"Failure rows: {int(data[TARGET].sum()):,}")

st.subheader("Enter operating measurements")
st.write("Choose values within the ranges found in the dataset.")

with st.form("prediction_form"):
    product_type = st.selectbox("Product type", sorted(data["Type"].unique()))
    left, right = st.columns(2)

    with left:
        air_temperature = st.number_input(
            "Air temperature [K]",
            min_value=float(data[NUMERIC_FEATURES[0]].min()),
            max_value=float(data[NUMERIC_FEATURES[0]].max()),
            value=float(data[NUMERIC_FEATURES[0]].mean()),
            step=0.1,
        )
        rotational_speed = st.number_input(
            "Rotational speed [rpm]",
            min_value=int(data[NUMERIC_FEATURES[2]].min()),
            max_value=int(data[NUMERIC_FEATURES[2]].max()),
            value=int(data[NUMERIC_FEATURES[2]].median()),
            step=1,
        )
        tool_wear = st.number_input(
            "Tool wear [min]",
            min_value=int(data[NUMERIC_FEATURES[4]].min()),
            max_value=int(data[NUMERIC_FEATURES[4]].max()),
            value=int(data[NUMERIC_FEATURES[4]].median()),
            step=1,
        )

    with right:
        process_temperature = st.number_input(
            "Process temperature [K]",
            min_value=float(data[NUMERIC_FEATURES[1]].min()),
            max_value=float(data[NUMERIC_FEATURES[1]].max()),
            value=float(data[NUMERIC_FEATURES[1]].mean()),
            step=0.1,
        )
        torque = st.number_input(
            "Torque [Nm]",
            min_value=float(data[NUMERIC_FEATURES[3]].min()),
            max_value=float(data[NUMERIC_FEATURES[3]].max()),
            value=float(data[NUMERIC_FEATURES[3]].median()),
            step=0.1,
        )

    submitted = st.form_submit_button("Check datapoint")

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "Type": product_type,
                "Air temperature [K]": air_temperature,
                "Process temperature [K]": process_temperature,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ],
        columns=FEATURES,
    )
    prediction = model.predict(input_data)[0]
    failure_probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.warning("The model predicts a failure for this datapoint.")
    else:
        st.success("The model predicts no failure for this datapoint.")

    st.metric("Estimated failure probability", f"{failure_probability:.1%}")
    st.caption(
        "This probability is the model's estimate for the selected values, "
        "not a guarantee or a prediction of a future failure."
    )
