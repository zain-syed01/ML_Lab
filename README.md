# AI4I 2020 Machine Failure Classification

A personal machine-learning project using the simulated AI4I 2020 Predictive Maintenance dataset. The notebook compares Logistic Regression and Random Forest; the Streamlit app lets you try operating measurements with a Random Forest model.

## Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository. It contains 10,000 simulated observations of machine operating conditions and failure labels.

Source: [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

## Run the notebook

Install the dependencies and open `main_notebook.ipynb` in Jupyter or VS Code. Keep `ai4i2020.csv` in the project folder beside the notebook.

```bash
pip install -r requirements.txt
```

## Run the dashboard

With the dataset and dashboard file in the project folder, run:

```bash
streamlit run streamlit_dashboard.py
```

The app trains a Random Forest on **all rows in the dataset** when it starts. This differs from the notebook, which holds out a test set to evaluate the models. Enter a product type and sensor measurements to get a datapoint classification and estimated failure probability. This is an educational demo using simulated data.
