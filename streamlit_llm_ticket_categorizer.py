
import streamlit as st
import pandas as pd
import json
import uuid # For generating unique IDs
from datetime import datetime # For ticket creation date
import os # For managing file paths

# --- Llama-cpp-python specific imports and setup ---
# You'll need to install these:
# pip install llama-cpp-python==0.1.85 --force-reinstall --no-cache-dir
# pip install huggingface_hub==0.20.3
try:
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
except ImportError:
    st.error("Please install llama-cpp-python and huggingface_hub: "
             "`pip install llama-cpp-python==0.1.85 --force-reinstall --no-cache-dir huggingface_hub==0.20.3`")
    st.stop() # Stop the app if libraries are not found

# --- Model Configuration (as per your notebook) ---
MODEL_NAME_OR_PATH = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_BASENAME = "mistral-7b-instruct-v0.2.Q6_K.gguf"

# --- Sample "Training" Tickets (from your notebook's final_data) ---
SAMPLE_TRAINING_TICKETS = [
    {
        "support_ticket_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "support_ticket_text": "My internet is not working. I can't connect to any websites.",
        "category": "Technical Issue",
        "tags": "internet, connectivity, network",
        "priority": "High",
        "eta": "1 Day",
        "response": "Thank you for reaching out. We understand your internet is not working. Our technical team is investigating this issue and aims to resolve it within 1 day. We appreciate your patience.",
        "ticket_create_date": "2024-05-20 10:00:00"
    },
    {
        "support_ticket_id": "b2c3d4e5-f6a7-8901-2345-67890abcdef0",
        "support_ticket_text": "I can't log in to my account. I keep getting an 'invalid credentials' error.",
        "category": "Technical Issue",
        "tags": "login, credentials, account",
        "priority": "High",
        "eta": "1 Day",
        "response": "We apologize for the login issue you're experiencing. Our team is looking into the 'invalid credentials' error and will provide an update within 1 day. Thank you for your patience.",
        "ticket_create_date": "2024-05-20 10:15:00"
    },
    {
        "support_ticket_id": "c3d4e5f6-a7b8-9012-3456-7890abcdef12",
        "support_ticket_text": "My laptop screen is flickering constantly, making it hard to work.",
        "category": "Hardware Issue",
        "tags": "screen, flickering, display, laptop",
        "priority": "High",
        "eta": "3 Days",
        "response": "We understand your laptop screen is flickering. This sounds like a hardware issue. Our team will assess the problem and aim for a resolution within 3 days. We appreciate your understanding.",
        "ticket_create_date": "2024-05-20 10:30:00"
    },
    {
        "support_ticket_id": "d4e5f6a7-b8c9-0123-4567-890abcdef34",
        "support_ticket_text": "I accidentally deleted a crucial file from my shared drive. Can it be recovered?",
        "category": "Data Recovery",
        "tags": "file, deleted, recovery, shared drive",
        "priority": "High",
        "eta": "3 Days",
        "response": "We understand the urgency of your accidentally deleted file. Our data recovery team is on it and will do their best to recover it within 3 days. We will keep you informed.",
        "ticket_create_date": "2024-05-20 10:45:00"
    },
    {
        "support_ticket_id": "e5f6a7b8-c9d0-1234-5678-90abcdef56",
        "support_ticket_text": "My printer is not responding. I've tried restarting it multiple times.",
        "category": "Hardware Issue",
        "tags": "printer, not responding, restart",
        "priority": "High",
        "eta": "3 Days",
        "response": "We're sorry to hear your printer isn't responding. Our team will investigate this hardware issue and aim for a resolution within 3 days. Thank you for your patience.",
        "ticket_create_date": "2024-05-20 11:00:00"
    },
    {
        "support_ticket_id": "f6a7b8c9-d0e1-2345-6789-0abcdef789",
        "support_ticket_text": "I can't access the company's internal wiki. It says 'access denied'.",
        "category": "Technical Issue",
        "tags": "access denied, wiki, internal",
        "priority": "High",
        "eta": "1 Day",
        "response": "We apologize for the access issue with the internal wiki. Our IT team is working to resolve this and restore your access within 1 day. We appreciate your understanding.",
        "ticket_create_date": "2024-05-20 11:15:00"
    },
    {
        "support_ticket_id": "a7b8c9d0-e1f2-3456-7890-abcdef9012",
        "support_ticket_text": "My mouse is not working on my desktop computer.",
        "category": "Hardware Issue",
        "tags": "mouse, desktop, not working",
        "priority": "High",
        "eta": "3 Days",
        "response": "We're sorry to hear your mouse isn't working. This appears to be a hardware issue. Our team will address it within 3 days. Thank you for your patience.",
        "ticket_create_date": "2024-05-20 11:30:00"
    },
    {
        "support_ticket_id": "b8c9d0e1-f2a3-4567-8901-234567890abc",
        "support_ticket_text": "I lost all my files on my external hard drive. It's not showing up.",
        "category": "Data Recovery",
        "tags": "files, external hard drive, lost, not showing",
        "priority": "High",
        "eta": "3 Days",
        "response": "We understand the concern about your lost files on the external hard drive. Our data recovery specialists will investigate and aim for recovery within 3 days. We'll keep you updated.",
        "ticket_create_date": "2024-05-20 11:45:00"
    },
    {
        "support_ticket_id": "c9d0e1f2-a3b4-5678-9012-34567890abcd",
        "support_ticket_text": "The software I need for my project keeps crashing. It's 'XYZ Software'.",
        "category": "Technical Issue",
        "tags": "software, crashing, XYZ Software",
        "priority": "High",
        "eta": "1 Day",
        "response": "We apologize for the issues with 'XYZ Software' crashing. Our technical team is aware of this and will work to resolve it within 1 day. Thank you for your patience.",
        "ticket_create_date": "2024-05-20 12:00:00"
    },
    {
        "support_ticket_id": "d0e1f2a3-b4c5-6789-0123-4567890abcde",
        "support_ticket_text": "My keyboard is not registering key presses on my workstation.",
        "category": "Hardware Issue",
        "tags": "keyboard, not registering, workstation",
        "priority": "High",
        "eta": "3 Days",
        "response": "We're sorry to hear your keyboard isn't working. This is a hardware issue that our team will address within 3 days. We appreciate your understanding.",
        "ticket_create_date": "2024-05-20 12:15:00"
    },
    {
        "support_ticket_id": "e1f2a3b4-c5d6-7890-1234-567890abcdef",
        "support_ticket_text": "I need to restore a previous version of a document from our cloud storage.",
        "category": "Data Recovery",
        "tags": "restore, document, cloud storage, previous version",
        "priority": "High",
        "eta": "3 Days",
        "response": "We can assist with restoring a previous version of your document from cloud storage. Our team will work on this and aim to complete it within 3 days. We'll keep you updated.",
        "ticket_create_date": "2024-05-20 12:30:00"
    },
    {
        "support_ticket_id": "f2a3b4c5-d6e7-8901-2345-67890abcdef0",
        "support_ticket_text": "The network drive is inaccessible. I can't open any files from it.",
        "category": "Technical Issue",
        "tags": "network drive, inaccessible, files",
        "priority": "High",
        "eta": "1 Day",
        "response": "We apologize for the inaccessible network drive. Our IT team is actively working to restore access and resolve this issue within 1 day. Thank you for your patience.",
        "ticket_create_date": "2024-05-20 12:45:00"
    },
    {
        "support_ticket_id": "a3b4c5d6-e7f8-9012-3456-7890abcdef12",
        "support_ticket_text": "My monitor is showing a black screen, but the computer is on.",
        "category": "Hardware Issue",
        "tags": "monitor, black screen, computer",
        "priority": "High",
        "eta": "3 Days",
        "response": "We're sorry to hear your monitor is showing a black screen. This is a hardware issue that our team will investigate and aim to resolve within 3 days. Thank you for your understanding.",
        "ticket_create_date": "2024-05-20 13:00:00"
    },
    {
        "support_ticket_id": "b4c5d6e7-f8a9-0123-4567-890abcdef34",
        "support_ticket_text": "I need to recover my old emails from an archived account.",
        "category": "Data Recovery",
        "tags": "emails, archived account, recover",
        "priority": "High",
        "eta": "3 Days",
        "response": "We can assist with recovering your old emails from the archived account. Our team will work on this and aim to complete it within 3 days. We'll keep you informed.",
        "ticket_create_date": "2024-05-20 13:15:00"
    },
    {
        "support_ticket_id": "c5d6e7f8-a9b0-1234-5678-90abcdef56",
        "support_ticket_text": "The company VPN is not connecting from my home office.",
        "category": "Technical Issue",
        "tags": "VPN, connecting, home office",
        "priority": "High",
        "eta": "1 Day",
        "response": "We apologize for the VPN connectivity issue from your home office. Our technical team is investigating and aims to resolve this within 1 day. Thank you for your patience.",
        "ticket_create_date": "2024-05-20 13:30:00"
    },
    {
        "support_ticket_id": "d6e7f8a9-b0c1-2345-6789-0abcdef789",
        "support_ticket_text": "My webcam is not detected during video calls.",
        "category": "Hardware Issue",
        "tags": "webcam, not detected, video calls",
        "priority": "High",
        "eta": "3 Days",
        "response": "We're sorry to hear your webcam isn't detected. This is a hardware issue that our team will investigate and aim to resolve within 3 days. We appreciate your understanding.",
        "ticket_create_date": "2024-05-20 13:45:00"
    },
    {
        "support_ticket_id": "e7f8a9b0-c1d2-3456-7890-abcdef9012",
        "support_ticket_text": "I need to restore my entire system from a backup.",
        "category": "Data Recovery",
        "tags": "system, restore, backup",
        "priority": "High",
        "eta": "3 Days",
        "response": "We understand the need to restore your system from a backup. Our data recovery team will assist you and aim to complete this within 3 days. We'll keep you informed.",
        "ticket_create_date": "2024-05-20 14:00:00"
    },
    {
        "support_ticket_id": "f8a9b0c1-d2e3-4567-8901-234567890abc",
        "support_ticket_text": "The company's main application is very slow today.",
        "category": "Technical Issue",
        "tags": "application, slow, performance",
        "priority": "High",
        "eta": "1 Day",
        "response": "We apologize for the slowness of the main application. Our technical team is actively investigating this performance issue and aims to resolve it within 1 day. Thank you for your patience.",
        "ticket_create_date": "2024-05-20 14:15:00"
    },
    {
        "support_ticket_id": "a9b0c1d2-e3f4-5678-9012-34567890abcd",
        "support_ticket_text": "My headset microphone is not working during online meetings.",
        "category": "Hardware Issue",
        "tags": "headset, microphone, online meetings",
        "priority": "High",
        "eta": "3 Days",
        "response": "We're sorry to hear your headset microphone isn't working. This is a hardware issue that our team will investigate and aim to resolve within 3 days. We appreciate your understanding.",
        "ticket_create_date": "2024-05-20 14:30:00"
    },
    {
        "support_ticket_id": "b0c1d2e3-f4a5-6789-0123-4567890abcde",
        "support_ticket_text": "I need to retrieve data from a corrupted USB drive.",
        "category": "Data Recovery",
        "tags": "data, corrupted, USB drive, retrieve",
        "priority": "High",
        "eta": "3 Days",
        "response": "We understand the need to retrieve data from your corrupted USB drive. Our data recovery team will work on this and aim to complete it within 3 days. We'll keep you informed.",
        "ticket_create_date": "2024-05-20 14:45:00"
    },
    {
        "support_ticket_id": "c1d2e3f4-a5b6-7890-1234-567890abcdef",
        "support_ticket_text": "The software update failed on my work computer.",
        "category": "Technical Issue",
        "tags": "software, update, failed, work computer",
        "priority": "Medium",
        "eta": "1 Day",
        "response": "We apologize for the failed software update on your work computer. Our technical team is investigating this and aims to resolve it within 1 day. Thank you for your patience.",
        "ticket_create_date": "2024-05-20 15:00:00"
    }
]

# --- Utility function to parse JSON output from the model ---
# This is adapted from your notebook's extract_json_data function.
def extract_json_data(json_str):
    try:
        # Find the indices of the opening and closing curly braces
        json_start = json_str.find('{')
        json_end = json_str.rfind('}')

        if json_start != -1 and json_end != -1:
            extracted_json = json_str[json_start:json_end + 1] # Extract the JSON object
            data_dict = json.loads(extracted_json)
            return data_dict
        else:
            st.warning(f"Warning: JSON object not found in response: {json_str}")
            return {}
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON from LLM: {e}. Raw output: {json_str}")
        return {}

# --- Load the Llama model using Streamlit's caching ---
# This decorator ensures the model is downloaded and loaded only once.
@st.cache_resource
def load_llama_model():
    st.info(f"Downloading {MODEL_BASENAME} from Hugging Face Hub...")
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_NAME_OR_PATH,
            filename=MODEL_BASENAME,
            cache_dir="./model_cache" # Cache directory for the model file
        )
        st.success("Model downloaded successfully!")
        st.info("Loading Llama model... This might take a moment.")
        llm = Llama(
            model_path=model_path,
            n_ctx=1024, # Context window size
            n_gpu_layers=-1, # Uncomment and set to a positive integer for GPU support (e.g., 30 for most layers)
                             # Set to -1 to offload all layers to GPU if possible.
            verbose=False # Suppress llama.cpp verbose output
        )
        st.success("Llama model loaded!")
        return llm
    except Exception as e:
        st.error(f"Failed to load Llama model: {e}")
        st.stop() # Stop the app if model loading fails

# Load the model globally (once)
llm_model = load_llama_model()

# --- LLM Interaction Function ---
def get_ticket_categorization_llama(description: str):
    """
    Calls the locally loaded Llama model to categorize a support ticket description,
    assign tags, priority, ETA, and a first response.
    Returns a dictionary with all the extracted information.
    """
    if not description.strip():
        return None

    # This prompt combines the logic from your notebook's Task 1, 2, 3, and 4
    # into a single, comprehensive request for the LLM.
    prompt = f"""
    [INST] You are an AI assistant for support ticket categorization.
    Analyze the following support ticket description and extract the following information:
    - **Category**: Classify the ticket into one of these types: 'Hardware Issue', 'Data Recovery', or 'Technical Issue'.
    - **Tags**: Provide a comma-separated list of relevant keywords or tags that describe the issue.
    - **Priority**: Assign a priority level: 'High', 'Medium', or 'Low'. Consider the urgency and impact.
    - **ETA**: Estimate the time to resolution (e.g., '1 Day', '3 Days', '24-48 hours', 'Immediate', 'Undetermined').
    - **Response**: Draft a concise, empathetic, professional, and helpful first response to the customer, acknowledging the issue and outlining next steps.

    Provide the output strictly in a JSON format. Ensure all curly braces are closed and there are no additional characters.

    Ticket Description: "{description}" [/INST]
    """

    try:
        with st.spinner("Categorizing ticket with Llama model..."):
            # Call the Llama model
            model_output = llm_model(
                prompt,
                max_tokens=512, # Increased max_tokens to allow for full response
                stop=["[INST]", "Q:", "\n"], # Stop sequences for Mistral-like models
                temperature=0.01, # Low temperature for deterministic output
                echo=False,
            )

            if model_output and model_output["choices"]:
                raw_output = model_output["choices"][0]["text"]
                # Use the extract_json_data utility function
                return extract_json_data(raw_output)
            else:
                st.error("Llama model did not return a valid response.")
                return None
    except Exception as e:
        st.error(f"Error during Llama model inference: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="LLM Support Ticket Categorizer", layout="wide")

st.title("🎫 LLM Support Ticket Categorizer (Local Llama Model)")
st.markdown("Enter a support ticket description below, and the AI will categorize it, assign tags, priority, ETA, and generate a first response using a local Mistral-7B model.")

# Initialize session state for tickets.
# This ensures that the sample tickets are only added once per Streamlit session.
if 'tickets' not in st.session_state:
    st.session_state.tickets = SAMPLE_TRAINING_TICKETS.copy()

# Input form for new ticket submission
with st.form("ticket_form"):
    support_ticket_text = st.text_area("Support Ticket Description", help="Describe the issue or request in detail.")
    submit_button = st.form_submit_button("Categorize Ticket")

    if submit_button:
        if support_ticket_text.strip():
            # Call the Llama model categorization function
            categorized_data = get_ticket_categorization_llama(support_ticket_text)
            if categorized_data:
                new_ticket = {
                    "support_ticket_id": str(uuid.uuid4()), # Auto-generate a unique ID for the new ticket
                    "support_ticket_text": support_ticket_text,
                    "category": categorized_data.get("category", "N/A"),
                    "tags": categorized_data.get("tags", "N/A"),
                    "priority": categorized_data.get("priority", "N/A"),
                    "eta": categorized_data.get("eta", "N/A"),
                    "response": categorized_data.get("response", "N/A"),
                    "ticket_create_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Record current timestamp
                }
                # Add the newly categorized ticket to the beginning of the list
                # so it appears at the top of the table.
                st.session_state.tickets.insert(0, new_ticket)
                st.success("Ticket categorized successfully!")
            else:
                st.warning("Could not categorize ticket. Please try a different description.")
        else:
            st.warning("Please enter a support ticket description.")

st.markdown("---")

# Display all tickets in a table format
st.subheader("All Support Tickets (Latest First)")
if st.session_state.tickets:
    df = pd.DataFrame(st.session_state.tickets)
    # Define the desired order of columns for the display table
    column_order = [
        "support_ticket_id",
        "support_ticket_text",
        "category",
        "tags",
        "priority",
        "eta",
        "response",
        "ticket_create_date"
    ]
    # Filter the DataFrame to include only the specified columns and maintain their order
    existing_columns = [col for col in column_order if col in df.columns]
    st.dataframe(df[existing_columns], use_container_width=True, height=500)
else:
    st.info("No tickets categorized yet. Enter a description above to get started!")

# Custom CSS for styling the Streamlit components to enhance aesthetics
st.markdown("""
<style>
    /* Styling for the primary submit button */
    .stButton>button {
        background-color: #4CAF50; /* Green background */
        color: white; /* White text */
        border-radius: 8px; /* Rounded corners for a modern look */
        padding: 10px 20px; /* Ample padding for better touch targets */
        font-size: 16px; /* Readable font size */
        border: none; /* No default border */
        cursor: pointer; /* Indicate interactivity */
        transition: background-color 0.3s ease; /* Smooth hover effect */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
    }
    .stButton>button:hover {
        background-color: #45a049; /* Slightly darker green on hover */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); /* Enhanced shadow on hover */
    }
    /* Styling for the text area input field */
    .stTextArea textarea {
        border-radius: 8px; /* Rounded corners */
        padding: 10px; /* Internal padding */
        border: 1px solid #ccc; /* Light grey border */
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05); /* Inner shadow for depth */
        font-family: 'Inter', sans-serif; /* Consistent font */
    }
    /* Styling for the DataFrame (table) display */
    .stDataFrame {
        border-radius: 8px; /* Rounded corners for the table container */
        overflow: hidden; /* Ensures content respects border-radius */
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08); /* Soft shadow for the table */
    }
    /* General body font */
    body {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)