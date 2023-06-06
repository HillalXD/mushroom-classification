import streamlit as st
import pickle
import numpy as np
import pandas as pd 

from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_path = "decissiontree.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define the mushroom classes
classes = ["edible", "poisonous"]

# Define the Streamlit app
def main():
    st.title("Mushroom Classification")
    st.write("This app classifies mushrooms as edible or poisonous.")

    # Gather user input
    gc = st.selectbox("select gill color", ["black", "brown", "buff", "chocolate", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"])
    spc = st.selectbox("spore print color", ["black", "brown", "buff", "chocolate", "green", "orange", "purple", "white", "yellow"])
    p = st.selectbox("population", ["abundant", "clustered", "numerous", "scattered", "several", "solitary"])
    gs = st.selectbox("gill size", ["broad", "narrow"])
    o = st.selectbox("odor", ["almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"])
    b = st.selectbox("bruises", ["bruises", "no"])
    ss = st.selectbox("stalk shape", ["enlarging", "tapering"])
    scar = st.selectbox("stalk color above ring", ["brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"])
    sr = st.selectbox("stalk root", ["bulbous", "club", "cup", "equal", "rhizomorphs", "rooted", "missing"])

    # Add more features here...

    # Prepare the input features
    features = np.array([gc, spc, p, gs, o, b, ss, scar, sr])
    df = pd.DataFrame([features])

    label = LabelEncoder()
    for col in df.columns:
        df[col] = label.fit_transform(df[col])


    # Make predictions
    prediction = model.predict(df)

    # Display the prediction
    st.write("Prediction:", classes[prediction[0]])

if __name__ == "__main__":
    main()
