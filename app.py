import streamlit as st
from PIL import Image
from datetime import date
import os

# Set page config
st.set_page_config(page_title="Moroccan Drought Segmentation", layout="wide")

# Load custom CSS for styling
with open("css/prediction.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="title"><h1>Moroccan Drought Segmentation</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="content-section"><p>This project uses deep learning models to predict future drought conditions in Morocco.</p></div>', unsafe_allow_html=True)

if 'prediction_date' not in st.session_state:
    st.session_state.prediction_date = None
if 'prediction_button_pressed' not in st.session_state:
    st.session_state.prediction_button_pressed = False
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = "Prediction"
if 'minimize_base_map' not in st.session_state:
    st.session_state.minimize_base_map = False

# Navigation buttons
st.markdown('<div class="top-bar">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)  # Increase to 4 columns for the new button

with col1:
    if st.button("Prediction", key="pred"):
        st.session_state.selected_section = "Prediction"
        st.session_state.minimize_base_map = True
with col2:
    if st.button("Model Metrics and History", key="model"):
        st.session_state.selected_section = "Model"
        st.session_state.minimize_base_map = False
with col3:
    if st.button("Data", key="data"):
        st.session_state.selected_section = "Data"
        st.session_state.minimize_base_map = False
with col4:
    if st.button("About Us", key="about"):
        st.session_state.selected_section = "About Us"
        st.session_state.minimize_base_map = False

st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.selected_section == "Prediction":
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.header("Prediction")
    
    # Date selection input
    st.session_state.prediction_date = st.date_input("Select Date for Prediction", 
                                                      value=st.session_state.get('prediction_date', None), 
                                                      key="prediction_date_input")
    
    # Display the "Prediction for Selected Date" button only if a date is selected
    if st.session_state.prediction_date:
        if st.button(f"Predict for {st.session_state.prediction_date}"):
            st.session_state.prediction_button_pressed = True
        if st.session_state.prediction_button_pressed:
            # Here you can call the model and show the prediction result
            st.write(f"Running prediction model for {st.session_state.prediction_date}...")

    # Display the base map image
    st.write("### Base Map")
    base_map_image = Image.open("BaseMap.png")
    
    # Columns for side-by-side images
    col1, col2 = st.columns([1, 1])  # Adjust ratio as needed
    
    # Base map in the first column
    with col1:
        if st.session_state.minimize_base_map:
            # Set width, height will adjust according to aspect ratio
            st.image(base_map_image, caption="Base Map of Morocco", width=800)
        else:
            st.image(base_map_image, caption="Base Map of Morocco", width=1000)

    # Predicted image in the second column
    with col2:
        if st.session_state.prediction_button_pressed:
            predicted_image = Image.open("predicted_image.png")
            if st.session_state.minimize_base_map:
                st.image(predicted_image, caption=f"Prediction for {st.session_state.prediction_date}", width=800)
            else:
                st.image(predicted_image, caption=f"Prediction for {st.session_state.prediction_date}", width=800)
        
elif st.session_state.selected_section == "Model":
    st.markdown('<div class="content-section model-section">', unsafe_allow_html=True)
    
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

    st.markdown('</div>', unsafe_allow_html=True)

    with open("css/model.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

elif st.session_state.selected_section == "Data":
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.write("This section provides information about the data used in this project.")

    # Date selection with custom validation
    selected_date = st.date_input(
        "Select a date to view the image:",
        min_value=date(2012, 1, 1),
        max_value=date(2024, 12, 31),
        value=date(2012, 1, 1),
        key="date_selector"
    )
    def is_valid_date(selected_date):
    # Check if the day is 1, 11, or 21
        return selected_date.day in [1, 11, 21]

    # Validate the selected date
    if is_valid_date(selected_date):
        formatted_date = selected_date.strftime('%Y-%m-%d')
        image_path = os.path.join('C:/Users/mlyou/Desktop/Master_DS/S3/DL/Project/Data/', f"{formatted_date}.png")
        
        if os.path.exists(image_path):
            st.image(image_path, caption=f"Image for {formatted_date}", use_container_width=True)
        else:
            st.write(f"No image available for {formatted_date}. Please choose another date.")
    else:
        st.write("Please select a valid date (1st, 11th, or 21st of any month).")

    st.markdown('</div>', unsafe_allow_html=True)
    

elif st.session_state.selected_section == "About Us":
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    with open("about_us.html", "r") as file:
        html_content = file.read()
        st.markdown(html_content, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    # Load custom CSS for styling
    with open("css/about_us.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
