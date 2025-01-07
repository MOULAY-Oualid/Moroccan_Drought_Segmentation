import streamlit as st
from PIL import Image
from datetime import date
import base64
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from inference import process_masks
import tempfile
from io import BytesIO
import os
import json


# Set page config
st.set_page_config(page_title="Moroccan Drought Segmentation", layout="wide")

# Folder ID from Google Drive 
folder_color_id = '1j5s1OnQI-b_Cc2NOYi6gEDydmGFBfH1q'
folder_gray_id = '1Hxh611o1QWTaTmRMahX6CjVDsWKKI7xV'

# Authenticate Google Drive using Streamlit secrets
def authenticate_google_drive():
    # Get the credentials from Streamlit secrets
    google_credentials = st.secrets["google"]["credentials"]
    
    # Convert the credentials string to a Python dictionary
    creds_dict = json.loads(google_credentials)
    
    # Create credentials using the loaded dictionary
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/drive'])
    
    # Build the Drive API client
    drive_service = build('drive', 'v3', credentials=creds)
    return drive_service


# Authenticate and get the service
service = authenticate_google_drive()

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


def find_nearest_dates(date_selected):
    # Split the input date string into year, month, and day
    year, month, day = map(int, date_selected.split('_'))

    # Possible days in a month
    possible_days = [1, 11, 21]

    nearest_dates = []

    # Check the current month for earlier possible days
    for possible_day in reversed(possible_days):
        if possible_day < day:
            nearest_dates.append(f"{year}_{month:02d}_{possible_day:02d}")

    # If we still need more dates, check the previous month
    if len(nearest_dates) < 2:
        if month == 1:
            prev_month, prev_year = 12, year - 1
        else:
            prev_month, prev_year = month - 1, year
        
        for possible_day in reversed(possible_days):
            nearest_dates.append(f"{prev_year}_{prev_month:02d}_{possible_day:02d}")
            if len(nearest_dates) == 2:
                break

    # Sort the dates chronologically and return the two nearest dates
    return sorted(nearest_dates)[:2]


# Function to upload a PNG image directly to a folder in Google Drive
def upload_image_to_drive(service, image, folder_id, file_name):
    # Convert the PIL image to a byte stream
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')  # Save the image as PNG
    image_bytes.seek(0)  # Go to the start of the byte stream

    # Set the metadata for the file upload
    file_metadata = {
        'name': file_name,  # File name
        'parents': [folder_id]  # The folder ID where you want to upload the file
    }

    # Create a MediaIoBaseUpload object for uploading the image
    media = MediaIoBaseUpload(image_bytes, mimetype='image/png')  # Set MIME type as PNG

    # Upload the image to Google Drive
    request = service.files().create(
        media_body=media,
        body=file_metadata
    )

    # Execute the upload and return the file's metadata
    uploaded_file = request.execute()

def process_and_upload_images(selected_date):
    # Format the selected date and find the nearest dates
    formatted_date = selected_date.strftime('%Y_%m_%d')
    nearest_date = find_nearest_dates(str(formatted_date))
    
    files_gray = list_all_files_in_drive(service, folder_gray_id)
    # Generate image names and fetch the corresponding files
    image_name1 = f"{nearest_date[0]}.png"
    image_name2 = f"{nearest_date[1]}.png"
    image_file1 = next((file for file in files_gray if file['name'] == image_name1), None)  
    image_file2 = next((file for file in files_gray if file['name'] == image_name2), None)

    # Retrieve image file ids
    file_id1 = image_file1['id']
    file_id2 = image_file2['id']
    
    # Fetch images from drive
    image1 = fetch_image_from_drive(service, file_id1)
    image2 = fetch_image_from_drive(service, file_id2)
    
    # Create a temporary directory for storing the images
    temp_dir = tempfile.mkdtemp(prefix="image_temp_")

    # Define paths for the images
    path1 = os.path.join(temp_dir, image_name1)
    path2 = os.path.join(temp_dir, image_name2)

    # Save image1 as PNG
    with open(path1, 'wb') as file:
        image1_bytes = BytesIO()
        image1.save(image1_bytes, format='PNG')
        image1_bytes.seek(0)
        file.write(image1_bytes.read())

    # Save image2 as PNG
    with open(path2, 'wb') as file:
        image2_bytes = BytesIO()
        image2.save(image2_bytes, format='PNG')
        image2_bytes.seek(0)
        file.write(image2_bytes.read())

    # Process the masks
    image_predicted, segmented_mask = process_masks([path1, path2])
    
    # Display the predicted image
    st.image(image_predicted, caption=f"Prediction for {formatted_date}", use_container_width=True)

    # Prepare file name and process the segmented mask
    file_name = f"{formatted_date}.png"
    segmented_mask = segmented_mask.convert('L')

    # Upload the segmented mask to the drive
    upload_image_to_drive(service, segmented_mask, folder_gray_id, file_name)



# Load custom CSS for styling at the start
with open("css/prediction.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
with open("css/model.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
with open("css/about_us.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



# Title and description
st.markdown('<div class="title"><h1>Moroccan Drought Segmentation</h1></div>', unsafe_allow_html=True)

# Manage session state
for key in ['prediction_date', 'prediction_button_pressed', 'selected_section']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'prediction_date' else False if key == 'prediction_button_pressed' else "Prediction"

# Use tabs for navigation
tabs = st.tabs(["Prediction", "Model Metrics and History", "Data", "About Us"])

with tabs[0]:
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    st.write("### For prediction, please select a date later than 2024-10-21.")
    c1, c2 , c3  = st.columns([1, 1 , 1 ])
    not_valid_date = False

    with c1:
        try:
            base_map_image = Image.open("assets/BaseMap_Morocco.png")
            st.image(base_map_image, caption="Base Map of Morocco", use_container_width=True)
        except FileNotFoundError:
            st.error("Base map image not found.")

    with c2:
        st.session_state.prediction_date = st.date_input(
            "Select Date for Prediction", 
            min_value=date(2012, 1, 1),
            value=date(2024, 11, 1), 
            key="prediction_date_input"
        )
    
        # Display the "Prediction for Selected Date" button only if a date is selected
        if st.session_state.prediction_date:
            selected_date = st.session_state.prediction_date
            if is_valid_date(selected_date):
                if selected_date > date(2024, 10, 21):
                    if st.button(f"Predict for {selected_date}"):
                        st.session_state.prediction_button_pressed = 'Predection'
                else:   
                    st.session_state.prediction_button_pressed = 'Data-from-drive'
            else:
                not_valid_date = True
                st.write("Please select a valid date (1st, 11th, or 21st of any month).")

    with c3:
        if st.session_state.prediction_button_pressed == 'Data-from-drive':
            formatted_date = selected_date.strftime('%Y_%m_%d')
            # List all files in the folder
            files = list_all_files_in_drive(service, folder_color_id)
            # Search for the image by date
            image_name = f"{formatted_date}.png"
            image_file = next((file for file in files if file['name'] == image_name), None)

            if image_file:
                # Fetch the image from Google Drive without downloading to disk
                file_id = image_file['id']
                image = fetch_image_from_drive(service, file_id)
                st.image(image, caption=f"Image from dataset for {formatted_date}", use_container_width=True)
                
        elif st.session_state.prediction_button_pressed == 'Predection' and not_valid_date == False: 
            try:
                date_selected = st.session_state.prediction_date
                process_and_upload_images(date_selected)
            except FileNotFoundError:
                st.error("Predicted image not found. Please ensure the prediction was successful.")
        elif not_valid_date == True:
            st.write(" ")
        else:   
            st.write(" ")
    st.markdown('</div>', unsafe_allow_html=True)
            
with tabs[1]:
    ccc1, ccc2 , ccc3 = st.columns([1, 1 , 1 ])
    with ccc2:
        # Example metrics - these should be actual metrics from your model training
        st.write("### Model Performance Metrics (ConvLSTM)")
        st.write("*Accuracy:* 92%")
        st.write("*Loss:* 0.91")
        # st.write("*Recall:* 91%")
        # st.write("*F1 Score:* 90%")
        results = Image.open("assets/results.jpg")
        st.image(results, caption="Model performance", width=800)    
    

# Display the image based on selected date
with tabs[2]:

    col1, col2  = st.columns([1, 1])
    with col1:
        selected_date = st.date_input(
            "Select a date to view the image:",
            min_value=date(2012, 1, 1),
            max_value=date(2024, 10, 21),
            value=date(2012, 1, 1),
            key="date_selector"
        )

    with col2:
        # List all files in the folder
        files = list_all_files_in_drive(service, folder_color_id)
        # Validate the selected date
        if is_valid_date(selected_date):
            formatted_date = selected_date.strftime('%Y_%m_%d')

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
