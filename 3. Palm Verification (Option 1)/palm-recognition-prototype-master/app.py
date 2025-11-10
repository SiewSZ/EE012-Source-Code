import streamlit as st
import numpy as np
import torch
import os
from PIL import Image
import tempfile
from client.cosine_similarity import calculate_similarity
from preprocess.preprocessor import preprocess_image
from self_supervised.model import PalmprintEncoder

st.set_page_config(page_title="Palmprint Authentication", layout="wide")

# --- CONFIGURATION ---
ENCODER_PATH = 'output/model/palmprint_encoder.pth'
REGISTERED_USERS_DIR = 'registered_users'
SIMILARITY_THRESHOLD = 0.8  # 80% similarity threshold for a match

# Create directory for registered users if it doesn't exist
os.makedirs(REGISTERED_USERS_DIR, exist_ok=True)


# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the pre-trained encoder model."""
    encoder = PalmprintEncoder().encoder
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=torch.device('cpu')))
    encoder.eval()
    return encoder


encoder = load_model()
device = torch.device('cpu')
encoder.to(device)


# --- HELPER FUNCTIONS ---
def get_embedding_and_roi(image_path):
    """Preprocesses an image, extracts its embedding, and returns the ROI."""
    preprocessed_roi = preprocess_image(image_path)
    if preprocessed_roi is None:
        return None, None

    # Prepare for model
    roi_for_model = np.expand_dims(preprocessed_roi, axis=0)
    roi_for_model = torch.from_numpy(roi_for_model).float().repeat(1, 3, 1, 1).to(device)

    with torch.no_grad():
        embedding = encoder(roi_for_model).cpu().numpy().flatten()
    return embedding, preprocessed_roi


def load_registered_users():
    """Loads all registered usernames and their embeddings."""
    users = {}
    for username in os.listdir(REGISTERED_USERS_DIR):
        user_dir = os.path.join(REGISTERED_USERS_DIR, username)
        if os.path.isdir(user_dir):
            embedding_path = os.path.join(user_dir, 'embedding.npy')
            if os.path.exists(embedding_path):
                embedding = np.load(embedding_path)
                users[username] = embedding
    return users


# --- UI PAGES ---
def page_register():
    """Page for registering a new user."""
    st.header("Register New User")
    username = st.text_input("Enter your username:")
    uploaded_file = st.file_uploader("Upload your palmprint image for registration", type=["jpg", "jpeg", "png", "tiff"])

    if st.button("Register"):
        if not username:
            st.warning("Please enter a username.")
        elif uploaded_file is None:
            st.warning("Please upload a palmprint image.")
        else:
            user_dir = os.path.join(REGISTERED_USERS_DIR, username)
            if os.path.exists(user_dir):
                st.error(f"Username '{username}' already exists. Please choose a different one.")
                return

            with st.spinner("Processing image and registering user..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name

                    embedding, preprocessed_roi = get_embedding_and_roi(temp_file_path)
                    os.remove(temp_file_path)

                    if embedding is not None and preprocessed_roi is not None:
                        os.makedirs(user_dir)
                        # Save embedding
                        np.save(os.path.join(user_dir, 'embedding.npy'), embedding)

                        # Save preprocessed ROI as a .npy file
                        np.save(os.path.join(user_dir, 'preprocessed.npy'), preprocessed_roi)

                        st.success(f"User '{username}' registered successfully!")

                        st.subheader("This is the Processed Image We Saved:")
                        # Display the ROI from the numpy array
                        st.image(preprocessed_roi, caption="Preprocessed Palmprint", width=200, clamp=True)
                    else:
                        st.error("Could not process the palmprint. Please try a different image.")
                except Exception as e:
                    st.error(f"An error occurred during registration: {e}")


def page_authenticate():
    """Page for authenticating a user."""
    st.header("Authenticate User")

    registered_users = load_registered_users()
    if not registered_users:
        st.warning("No users are registered in the system yet. Please register first.")
        return

    # 1. Enter Username
    entered_username = st.text_input("Enter your username to verify:")

    # 2. Upload Palmprint
    uploaded_file = st.file_uploader("Upload your palmprint image to authenticate", type=["jpg", "jpeg", "png", "tiff"])

    # 3. Authenticate Button
    if st.button("Authenticate"):
        if not entered_username:
            st.warning("Please enter your username.")
            return
        if entered_username not in registered_users:
            st.error("This username is not registered. Please check the spelling or register first.")
            return
        if uploaded_file is None:
            st.warning("Please upload your palmprint image for authentication.")
            return

        with st.spinner("Verifying..."):
            try:
                # Get the embedding of the uploaded image
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                query_embedding, _ = get_embedding_and_roi(temp_file_path)
                os.remove(temp_file_path)

                if query_embedding is None:
                    st.error("Could not process the uploaded image. Please try again.")
                    return

                # Get the registered embedding for the entered user
                registered_embedding = registered_users[entered_username]

                # 4. Compare and give result
                similarity_score = calculate_similarity(query_embedding.reshape(1, -1), registered_embedding.reshape(1, -1))[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Your Uploaded Palmprint")
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)

                with col2:
                    st.subheader("Authentication Result")
                    if similarity_score >= SIMILARITY_THRESHOLD:
                        st.success(f"Authentication Successful! Welcome, {entered_username}.")
                        st.markdown(f"**Verification Score:** `{similarity_score * 100:.2f}%`")
                    else:
                        st.error("Authentication Failed. The palmprint does not match the registered user.")
                        st.markdown(f"**Verification Score:** `{similarity_score * 100:.2f}%` (below threshold).")

                    # Display the registered image for comparison
                    registered_image_path = os.path.join(REGISTERED_USERS_DIR, entered_username, 'preprocessed.npy')
                    if os.path.exists(registered_image_path):
                        st.subheader("Your Registered Palmprint:")
                        registered_image = np.load(registered_image_path)
                        st.image(registered_image, caption=f"Registered print for {entered_username}", use_container_width=True, clamp=True)

            except Exception as e:
                st.error(f"An error occurred during authentication: {e}")


# --- MAIN APP LOGIC ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Register", "Authenticate"])

if page == "Register":
    page_register()
elif page == "Authenticate":
    page_authenticate()