LLM-Powered Support Ticket Categorization App
This project implements a Streamlit web application for automatically categorizing support tickets, assigning relevant tags, determining priority and estimated time to resolution (ETA), and generating a draft response using a local Large Language Model (LLM). The application is designed to streamline the initial triage process for support teams.

✨ Features
Automated Categorization: Classifies support tickets into predefined categories (e.g., Technical Issue, Hardware Issue, Data Recovery).

Intelligent Tagging: Assigns relevant keywords/tags based on the ticket description.

Priority & ETA Assignment: Determines the urgency and estimated resolution time.

Draft Response Generation: Generates an empathetic and professional first response to the customer.

Interactive UI: User-friendly interface built with Streamlit.

Historical View: Displays all categorized tickets in a sortable table, with the latest at the top, including sample "training" data.

Local LLM Inference: Utilizes llama-cpp-python to run the LLM locally on your machine, offering privacy and potentially lower latency compared to API-based solutions.

🚀 Technologies Used
Python 3.x

Streamlit: For building the interactive web application.

llama-cpp-python: Python bindings for llama.cpp, enabling efficient local inference of GGUF models.

huggingface_hub: For downloading the LLM model from Hugging Face Model Hub.

Mistral-7B-Instruct-v0.2 (GGUF): The specific Large Language Model used for categorization and response generation.

Pandas: For data handling and table display.

uuid & datetime: For generating unique IDs and timestamps.


🧠 LLM Model Details
This application uses the Mistral-7B-Instruct-v0.2 model, specifically a 6-bit quantized version (mistral-7b-instruct-v0.2.Q6_K.gguf). This model is run locally using the llama-cpp-python library, which provides efficient inference capabilities for GGUF-formatted models on various hardware configurations (CPU and GPU).

