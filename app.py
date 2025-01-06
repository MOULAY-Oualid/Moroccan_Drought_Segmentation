import streamlit as st
from PIL import Image
from datetime import date
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import json

# Authenticate using the service account
def authenticate_google_drive():
    # Load the JSON credentials from Streamlit secrets
    credentials_json = st.secrets["google"]["credentials"]
    
    # Parse the JSON string into a Python dictionary
    service_account_info = json.loads(credentials_json)
    
    # Create credentials object
    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    
    # Build the Google Drive service
    service = build('drive', 'v3', credentials=creds)
    return service

# List all files in Google Drive folder (with pagination)
def list_all_files_in_drive(service, folder_id):
    files = []
    page_token = None
    while True:
        # Retrieve files in the folder with pagination
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()
        
        # Append the files to the list
        files.extend(results.get('files', []))
        
        # Check if there is another page of files
        page_token = results.get('nextPageToken')
        if not page_token:
            break  # No more pages, exit the loop
    
    return files

# Folder ID from Google Drive (replace with your actual folder ID)
folder_id = '1j5s1OnQI-b_Cc2NOYi6gEDydmGFBfH1q'

# Authenticate and get the service
service = authenticate_google_drive()

# List all files in the folder
files = list_all_files_in_drive(service, folder_id)

# Function to fetch image from Google Drive and return as PIL Image
def fetch_image_from_drive(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()  # Use BytesIO to store image data in memory
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)  # Go to the start of the file
    image = Image.open(fh)  # Open the image from memory
    return image

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


# Load custom CSS for styling at the start
with open("css/prediction.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
with open("css/model.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
with open("css/about_us.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Set page config
st.set_page_config(page_title="Moroccan Drought Segmentation", layout="wide")

# Title and description
st.markdown('<div class="title"><h1>Moroccan Drought Segmentation</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="content-section"><p>This project uses deep learning models to predict future drought conditions in Morocco.</p></div>', unsafe_allow_html=True)

# Manage session state
for key in ['prediction_date', 'prediction_button_pressed', 'selected_section']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'prediction_date' else False if key == 'prediction_button_pressed' else "Prediction"

# Use tabs for navigation
tabs = st.tabs(["Prediction", "Model Metrics and History", "Data", "About Us"])

with tabs[0]:
    st.markdown('<div class="container">', unsafe_allow_html=True)
    c1, c2 , c3  = st.columns([1, 1 , 1 ])

    with c1:
        try:
            base_map_image = Image.open("assets/BaseMap_Morocco.png")
            st.image(base_map_image, caption="Base Map of Morocco", use_container_width=True)
        except FileNotFoundError:
            st.error("Base map image not found.")

    with c2:
        st.session_state.prediction_date = st.date_input("Select Date for Prediction", value=st.session_state.get('prediction_date', None), key="prediction_date_input")
    
        # Display the "Prediction for Selected Date" button only if a date is selected
        if st.session_state.prediction_date:
            if st.button(f"Predict for {st.session_state.prediction_date}"):
                st.session_state.prediction_button_pressed = True

    with c3:
        if st.session_state.prediction_button_pressed:
            try:
                predicted_image = Image.open("assets/predicted_image.png")
                st.image(predicted_image, caption=f"Prediction for {st.session_state.prediction_date}", use_container_width=True)
            except FileNotFoundError:
                st.error("Predicted image not found. Please ensure the prediction was successful.")
    st.markdown('</div>', unsafe_allow_html=True)
            
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
    

# Display the image based on selected date
with tabs[2]:

    col1, col2  = st.columns([1, 1])
    with col1:
        selected_date = st.date_input(
            "Select a date to view the image:",
            min_value=date(2012, 1, 1),
            max_value=date(2024, 12, 31),
            value=date(2012, 1, 1),
            key="date_selector"
        )

    with col2:
        # Validate the selected date
        if is_valid_date(selected_date):
            formatted_date = selected_date.strftime('%Y-%m-%d')

            # Search for the image by date
            image_name = f"{formatted_date}.png"
            image_file = next((file for file in files if file['name'] == image_name), None)

            if image_file:
                # Fetch the image from Google Drive without downloading to disk
                file_id = image_file['id']
                image = fetch_image_from_drive(service, file_id)
                st.image(image, caption=f"Image for {formatted_date}", use_container_width=True)
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
