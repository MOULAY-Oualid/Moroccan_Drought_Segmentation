import streamlit as st
from PIL import Image
from datetime import date
import os
import base64

# Set page config
st.set_page_config(page_title="Moroccan Drought Segmentation", layout="wide")

# Load custom CSS for styling at the start
with open("css/prediction.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
with open("css/model.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
with open("css/about_us.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="title"><h1>Moroccan Drought Segmentation</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="content-section"><p>This project uses deep learning models to predict future drought conditions in Morocco.</p></div>', unsafe_allow_html=True)

# Manage session state
for key in ['prediction_date', 'prediction_button_pressed', 'selected_section', 'minimize_base_map']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'prediction_date' else False if key in ['prediction_button_pressed', 'minimize_base_map'] else "Prediction"

# Use tabs for navigation
tabs = st.tabs(["Prediction", "Model Metrics and History", "Data", "About Us"])

def is_valid_date(selected_date):
    # Check if the day is 1, 11, or 21
    return selected_date.day in [1, 11, 21]

def img_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
        return None

with tabs[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.session_state.prediction_date = st.date_input("Select Date for Prediction", value=st.session_state.get('prediction_date', None), key="prediction_date_input")
    
    # Display the "Prediction for Selected Date" button only if a date is selected
    if st.session_state.prediction_date:
        if st.button(f"Predict for {st.session_state.prediction_date}"):
            st.session_state.prediction_button_pressed = True
        if st.session_state.prediction_button_pressed:
            st.write(f"Running prediction model for {st.session_state.prediction_date}...")

    # Display the base map image
    st.write("### Base Map")
    try:
        base_map_image = Image.open("BaseMap.png")
        st.image(base_map_image, caption="Base Map of Morocco", use_container_width=True)
    except FileNotFoundError:
        st.error("Base map image not found.")

    if st.session_state.prediction_button_pressed:
        try:
            predicted_image = Image.open("predicted_image.png")
            st.write(f"### Prediction Result for {st.session_state.prediction_date}")
            st.image(predicted_image, caption=f"Prediction for {st.session_state.prediction_date}", use_container_width=True)
        except FileNotFoundError:
            st.error("Predicted image not found. Please ensure the prediction was successful.")
            
with tabs[1]:
    # Example metrics - these should be actual metrics from your model training
    st.subheader("Model Performance Metrics")
    st.write("**Accuracy:** 92%")
    st.write("**Precision:** 89%")
    st.write("**Recall:** 91%")
    st.write("**F1 Score:** 90%")

    # Displaying model history (e.g., loss over epochs)
    st.subheader("Training History")
    epochs = list(range(1, 21))  # assuming 20 epochs
    loss = [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06]

    # Since you cannot execute code to plot, here's a conceptual description:
    st.write("Graph would show:")
    st.write(f"- X-axis: Epochs {epochs}")
    st.write(f"- Y-axis: Loss {loss}")
    st.write("Decreasing trend indicating model improvement over epochs.")

    # Additional information or plots can be added here, like confusion matrix or ROC curve
    st.write("### Additional Model Information")
    st.write("Here you could add more detailed information or visualizations like:")
    st.write("- Confusion Matrix")
    st.write("- ROC Curve")
    st.write("- Learning Rate Schedule")


with tabs[2]:
    st.write("This section provides information about the data used in this project.")

    col1, col2 = st.columns([1, 1])
    with col1:
        selected_date = st.date_input(
            "Select a date to view the image:",
            min_value=date(2012, 1, 1),
            max_value=date(2024, 12, 31),
            value=date(2012, 1, 1),
            key="date_selector"
        )
    
    # Validate the selected date
    if is_valid_date(selected_date):
        formatted_date = selected_date.strftime('%Y-%m-%d')
        image_path = os.path.join('./Data/', f"{formatted_date}.png")
        
        if os.path.exists(image_path):
            st.image(image_path, caption=f"Image for {formatted_date}", use_container_width=True)
        else:
            st.write(f"No image available for {formatted_date}. Please choose another date.")
    else:
        st.write("Please select a valid date (1st, 11th, or 21st of any month).")
    

with tabs[3]:
    # Convert images to Base64
    images = {
        "oualid": img_to_base64("assets/oualid.png"),
        "ahmed": img_to_base64("assets/ahmed.png"),
        "youssef": img_to_base64("assets/youssef.png"),
        "moad": img_to_base64("assets/moad.png"),
        "linkedin-icon": img_to_base64("assets/linkedin-icon.png"),
        "github-icon": img_to_base64("assets/github-icon.png")
    }

    # Read HTML content
    try:
        with open("about_us.html", "r") as file:
            html_content = file.read()
        for name, base64_str in images.items():
            if base64_str:
                html_content = html_content.replace(f'assets/{name}.png', f'data:image/png;base64,{base64_str}')
        
        st.markdown(html_content, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Failed to load the 'About Us' content.")