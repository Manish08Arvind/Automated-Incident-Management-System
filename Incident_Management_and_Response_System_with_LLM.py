import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
import json
import random
from sklearn.metrics import accuracy_score

!rm -rf ~/.cache/huggingface
print("Hugging Face cache cleared.")

# Installation for GPU llama-cpp-python.
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.85 --force-reinstall --no-cache-dir -q

from llama_cpp import Llama

from huggingface_hub import hf_hub_download

# Mounting google drive to access files.
from google.colab import drive
drive.mount('/content/drive')

# file path.
path = '/content/drive/MyDrive/Support_ticket_text_data_mid_term.csv'

data = pd.read_csv(path)

print(f"Generated synthetic dataset with {len(data)} records.")
print(data.head())

# --- 1. Utility functions (from original script) ---
def extract_json_data(json_str):
    """Defining a function to parse the JSON output from the model."""
    try:
        json_start = json_str.find('{')
        json_end = json_str.rfind('}')
        if json_start != -1 and json_end != -1:
            extracted_category = json_str[json_start:json_end + 1]
            data_dict = json.loads(extracted_category)
            return data_dict
        else:
            # print(f"Warning: JSON object not found in response: {json_str}")
            return {}
    except json.JSONDecodeError as e:
        # print(f"Error parsing JSON: {e} in {json_str}")
        return {}

# --- 2. Define Models and Their Paths ---
models_to_test = {
    "Mistral-7B-Instruct-v0.2": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q6_K.gguf",
        "llm_instance": None
    },
    "Llama-2-7B-Chat": {
        "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
        "filename": "llama-2-7b-chat.Q6_K.gguf",
        "llm_instance": None
    },
    "TinyLlama-1.1B-Chat": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
        "llm_instance": None
    },
    "OpenHermes-2.5-Mistral-7B": {
        "repo_id": "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
        "filename": "openhermes-2.5-mistral-7b.Q6_K.gguf",
        "llm_instance": None
    }
}

print("Downloading models (this may take some time)...")
for model_name, details in models_to_test.items():
    print(f"Downloading {model_name}...")
    try:
        model_path = hf_hub_download(repo_id=details["repo_id"], filename=details["filename"])
        models_to_test[model_name]["llm_instance"] = Llama(model_path=model_path, n_ctx=1024, verbose=True)
        print(f"{model_name} loaded successfully.")
    except BaseException as e:
        print(f"Failed to load {model_name}: {e}")
        models_to_test[model_name]["llm_instance"] = None

# Filter out models that failed to load
models_to_test = {k: v for k, v in models_to_test.items() if v["llm_instance"] is not None}
if not models_to_test:
    print("No models were loaded successfully. Exiting.")
    exit()

# --- 3. Prompt Definitions (from original script) ---
prompt_1 = """
   As an AI, your job is to categorize IT support tickets.
   Please label each ticket as either a Hardware Issue, Data Recovery, or Technical Issue.
   Your response should be in the format: {"category": "Hardware Issues"}, {"category": "Data Recovery"}, or {"category": "Technical Issues"}.
   Keep your output simple and accurate. Ensure that all curly braces are closed and there are no additional characters in the output.
"""

prompt_2 = """
   As an AI, your task is to label IT support tickets with relevant tags.
   Please identify the most appropriate keywords and include them in your response.
   Your output should be formatted as follows: {"tags": ["Wifi", "Data Loss", "Connection Issues", "Battery"]}.
   Keep your output simple and accurate. Ensure that all curly braces are closed and there are no additional characters in the output.
"""

prompt_3 = """
    As an AI, your task is to determine the priority and estimated time to resolve (ETA) for IT support tickets.
    Consider the severity of the issue, the time needed for resolution, and customer satisfaction.
    Your response should be in the format: {"priority": "High", "eta": "2 Days"}.
    Keep your output simple and accurate. Ensure that all curly braces are closed and there are no additional characters in the output.
"""

prompt_4 = """
    As an AI, your task is to draft a response for IT support tickets.
    Consider customer satisfaction, the severity of the issue, and the company's responsibility.
    Your response should be in the format: {"response": "This is a draft response"}.
    Ensure your response is empathetic, professional, helpful, and concise.
    Please ensure that all curly braces are closed and there are no additional characters in the output.
"""

# --- 4. Response Functions (from original script, slightly modified for flexibility) ---
def get_model_response(llm_instance, prompt, ticket, category=None, tags=None, priority=None, eta=None, max_tokens=1024, is_json=True):
    full_prompt = f"Q: {prompt}\nSupport ticket: {ticket}"
    if category: full_prompt += f"\nCategory: {category}"
    if tags: full_prompt += f"\nTags: {tags}"
    if priority: full_prompt += f"\nPriority: {priority}"
    if eta: full_prompt += f"\nETA: {eta}"
    full_prompt += "\nA:"

    model_output = llm_instance(
        full_prompt,
        max_tokens=max_tokens,
        stop=["Q:", "\n"],
        temperature=0.01,
        echo=False,
    )
    temp_output = model_output["choices"][0]["text"]
    if is_json:
        # Attempt to find JSON, if not found, return full output for error handling
        json_start = temp_output.find('{')
        json_end = temp_output.rfind('}')
        if json_start != -1 and json_end != -1:
            return temp_output[json_start:json_end + 1]
        else:
            return temp_output # Return full output if JSON markers not found
    return temp_output

# --- 5. Generate Synthetic Ground Truth for Comparison ---
# This is a simplified approach. In a real scenario, these would be human-labeled.
# We'll assign labels based on keywords, which might not be perfect, but serves the purpose
# of having a comparison baseline for synthetic data.
def assign_ground_truth(ticket_text):
    text_lower = ticket_text.lower()
    true_category = "Technical Issues"
    true_priority = "Medium"
    true_eta = "1 Day"
    true_tags = []

    if "laptop" in text_lower or "computer" in text_lower or "hardware" in text_lower or "printer" in text_lower or "monitor" in text_lower or "keyboard" in text_lower:
        true_category = "Hardware Issues"
        if "not turning on" in text_lower or "crashing" in text_lower:
            true_priority = "High"
            true_eta = "3 Days"
        true_tags.append("Device")
        if "printer" in text_lower: true_tags.append("Printer")
        if "battery" in text_lower: true_tags.append("Battery")
        if "screen" in text_lower: true_tags.append("Screen")

    if "recover" in text_lower or "deleted files" in text_lower or "lost data" in text_lower:
        true_category = "Data Recovery"
        true_priority = "High"
        true_eta = "3 Days"
        true_tags.append("Data Loss")
        true_tags.append("Files")

    if "network" in text_lower or "wifi" in text_lower or "internet" in text_lower or "connection" in text_lower or "login" in text_lower or "password" in text_lower or "software" in text_lower or "application" in text_lower or "email" in text_lower or "vpn" in text_lower:
        if "network" in text_lower or "wifi" in text_lower or "connection" in text_lower:
            true_tags.append("Network")
            true_tags.append("Connection Issues")
        if "password" in text_lower or "login" in text_lower or "account" in text_lower:
            true_tags.append("Account")
            true_tags.append("Login Issues")
            true_priority = "High" # Account access often high priority
            true_eta = "1 Day"
        if "software" in text_lower or "application" in text_lower:
            true_tags.append("Software")
            true_tags.append("Installation")
        if "email" in text_lower: true_tags.append("Email")
        if "vpn" in text_lower: true_tags.append("VPN")

    # Ensure at least one tag if specific ones weren't found
    if not true_tags:
        true_tags.append("General Tech")

    return {
        'true_category': true_category,
        'true_tags': list(set(true_tags)), # Remove duplicates
        'true_priority': true_priority,
        'true_eta': true_eta
    }

print("\nGenerating synthetic ground truth...")
ground_truth_df = data['support_ticket_text'].apply(assign_ground_truth).apply(pd.Series)
data = pd.concat([data, ground_truth_df], axis=1)
print("Synthetic ground truth generated.")
print(data.head())

# --- 6. Run Tasks for Each Model and Store Results ---
results_by_model = {}

for model_name, details in models_to_test.items():
    if details["llm_instance"] is None:
        continue

    print(f"\n--- Running tasks for Model: {model_name} ---")
    current_llm = details["llm_instance"]
    model_data = data.copy() # Use a copy for each model's predictions
    processing_times = {}

    # Task 1: Categorization
    print("Task 1: Categorization...")
    start_time = time.time()
    model_data['model_response_cat'] = model_data['support_ticket_text'].apply(
        lambda x: get_model_response(current_llm, prompt_1, x, max_tokens=10, is_json=True)
    )
    model_data['parsed_category'] = model_data['model_response_cat'].apply(extract_json_data).apply(lambda x: x.get('category'))
    processing_times['Task1_Categorization'] = time.time() - start_time
    print(f"Time taken for Categorization: {processing_times['Task1_Categorization']:.2f} seconds")

    # Task 2: Tagging
    print("Task 2: Tagging...")
    start_time = time.time()
    model_data['model_response_tags'] = model_data.apply(
        lambda row: get_model_response(current_llm, prompt_2, row['support_ticket_text'],
                                       category=row['parsed_category'], max_tokens=1024, is_json=True), axis=1
    )
    model_data['parsed_tags'] = model_data['model_response_tags'].apply(extract_json_data).apply(lambda x: x.get('tags'))
    processing_times['Task2_Tagging'] = time.time() - start_time
    print(f"Time taken for Tagging: {processing_times['Task2_Tagging']:.2f} seconds")

    # Task 3: Priority & ETA
    print("Task 3: Priority & ETA...")
    start_time = time.time()
    model_data['model_response_priority_eta'] = model_data.apply(
        lambda row: get_model_response(current_llm, prompt_3, row['support_ticket_text'],
                                       category=row['parsed_category'], tags=row['parsed_tags'],
                                       max_tokens=20, is_json=True), axis=1
    )
    parsed_prio_eta = model_data['model_response_priority_eta'].apply(extract_json_data)
    model_data['parsed_priority'] = parsed_prio_eta.apply(lambda x: x.get('priority'))
    model_data['parsed_eta'] = parsed_prio_eta.apply(lambda x: x.get('eta'))
    processing_times['Task3_Priority_ETA'] = time.time() - start_time
    print(f"Time taken for Priority & ETA: {processing_times['Task3_Priority_ETA']:.2f} seconds")

    # Task 4: Draft Response
    print("Task 4: Draft Response...")
    start_time = time.time()
    model_data['model_response_draft'] = model_data.apply(
        lambda row: get_model_response(current_llm, prompt_4, row['support_ticket_text'],
                                       category=row['parsed_category'], tags=row['parsed_tags'],
                                       priority=row['parsed_priority'], eta=row['parsed_eta'],
                                       max_tokens=1024, is_json=False), axis=1
    )
    # For draft response, we expect a plain string output, so no json parsing to 'parsed_response' column
    # model_data['parsed_response'] = model_data['model_response_draft'].apply(extract_json_data).apply(lambda x: x.get('response'))
    processing_times['Task4_Draft_Response'] = time.time() - start_time
    print(f"Time taken for Draft Response: {processing_times['Task4_Draft_Response']:.2f} seconds")


    results_by_model[model_name] = {
        "predictions_df": model_data,
        "processing_times": processing_times
    }
    print(f"Finished processing for {model_name}.")

# --- 7. Performance Comparison ---
print("\n--- Performance Comparison ---")

comparison_metrics = {}

for model_name, res in results_by_model.items():
    predictions_df = res["predictions_df"]

    # Calculate Accuracy for Classification Tasks
    try:
        accuracy_category = accuracy_score(predictions_df['true_category'], predictions_df['parsed_category'].fillna(''))
        accuracy_priority = accuracy_score(predictions_df['true_priority'], predictions_df['parsed_priority'].fillna(''))
        accuracy_eta = accuracy_score(predictions_df['true_eta'], predictions_df['parsed_eta'].fillna(''))

        comparison_metrics[model_name] = {
            "Category Accuracy": accuracy_category,
            "Priority Accuracy": accuracy_priority,
            "ETA Accuracy": accuracy_eta,
            "Processing Times (s)": res["processing_times"]
        }
    except Exception as e:
        print(f"Could not calculate metrics for {model_name} due to error: {e}")
        comparison_metrics[model_name] = {
            "Error": str(e),
            "Processing Times (s)": res["processing_times"]
        }

# Display Comparison Table
comparison_df = pd.DataFrame.from_dict(comparison_metrics, orient='index')
print("\nAccuracy and Processing Times Comparison:")
print(comparison_df.round(4))

# --- Qualitative Comparison Framework ---
print("\n--- Qualitative Comparison (Manual Review Required) ---")
print("Below are sample outputs for each model. Please review them manually for quality:")

sample_indices = random.sample(range(len(data)), min(5, len(data))) # Get up to 5 random samples

for i in sample_indices:
    print(f"\n--- Original Ticket ID: {data.loc[i, 'support_tick_id']} ---")
    print(f"Ticket Text: {data.loc[i, 'support_ticket_text']}")
    print(f"Ground Truth Category: {data.loc[i, 'true_category']}")
    print(f"Ground Truth Tags: {data.loc[i, 'true_tags']}")
    print(f"Ground Truth Priority: {data.loc[i, 'true_priority']}")
    print(f"Ground Truth ETA: {data.loc[i, 'true_eta']}")

    for model_name, res in results_by_model.items():
        if model_name in comparison_metrics and "Error" not in comparison_metrics[model_name]: # Only show if model loaded
            print(f"\n  -- Model: {model_name} --")
            print(f"    Predicted Category: {res['predictions_df'].loc[i, 'parsed_category']}")
            print(f"    Predicted Tags: {res['predictions_df'].loc[i, 'parsed_tags']}")
            print(f"    Predicted Priority: {res['predictions_df'].loc[i, 'parsed_priority']}")
            print(f"    Predicted ETA: {res['predictions_df'].loc[i, 'parsed_eta']}")
            print(f"    Draft Response: {res['predictions_df'].loc[i, 'model_response_draft']}")

