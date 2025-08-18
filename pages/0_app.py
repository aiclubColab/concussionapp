from embeddings import get_openai_embeddings
import streamlit as st
import pandas as pd
import numpy as np
import pickle

MODEL_NAME = "balanced_MLP_best_model"

@st.cache_resource
def load_model(model_name):
    with open(model_name, "rb") as file_name:
        return pickle.load(file_name)

# Load the model
tabular_model = load_model(MODEL_NAME)

# Title
st.title("Soccer Concussion Classification")
st.subheader("User Dashboard")

# Initialize session state for the inputs
for key in ["age", "gender", "issue", "body_part_affected"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# Input fields
st.session_state.age = st.text_input("Enter Age", st.session_state.age)
st.session_state.gender = st.text_input("Enter The Gender", st.session_state.gender)
# st.session_state.gender = st.selectbox("Enter The Gender", ("Male", "Female"), index=0 if st.session_state.gender == "" else ("Male", "Female").index(st.session_state.gender))
st.session_state.issue = st.text_input("Issue / Accident Happened", st.session_state.issue)
st.session_state.body_part_affected = st.text_input("Body Part Affected", st.session_state.body_part_affected)

# Apply button
if st.button("Apply"):
    if not st.session_state.age or not st.session_state.gender or not st.session_state.issue or not st.session_state.body_part_affected:
        st.error("Please enter all required information")
    else:
        st.success("All information provided!")
        text  = f"{st.session_state.age} Year Old {st.session_state.gender} PLAYING SOCCER, {st.session_state.issue}. Body part affected {st.session_state.body_part_affected}"
        # User Data
        st.subheader("User Data")
        st.divider()
        st.write(text)
        st.divider()
    
        # Generate embeddings
        embeddings = get_openai_embeddings(text)

        # Ensure the embeddings are in 2D shape for the model
        embedding_array = np.array(embeddings).reshape(1, -1)

        # Make prediction
        index = tabular_model.predict(embedding_array)
        labels = ['Concussion', 'No Concussion']
        injury_status = labels[index[0]]

        # Display prediction in Streamlit
        st.subheader("Predictions")
        st.write(f"**{injury_status}**")


# Clear All button (outside Apply block)
if st.button("Clear All"):
    for key in ["age", "gender", "issue", "body_part_affected"]:
        st.session_state[key] = ""
    st.rerun()