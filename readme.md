# Langchain Tool-Based Chatbot

This project implements an interactive chatbot using Langchain, LangGraph, and several external tools like Arxiv and Wikipedia. The chatbot can respond to user queries by either using an AI assistant (Groq-based language model) or fetching relevant data from research tools like Arxiv and Wikipedia. The application is built using Streamlit for the web interface.

## Features

- **Interactive Chat**: Communicate with a chatbot that uses both AI and research tools for responses.
- **Research Tools**: The bot can fetch data from Arxiv and Wikipedia, providing well-rounded and informed answers.
- **Source Labeling**: Each response includes a label to indicate whether it comes from the Assistant or from external tools (Arxiv/Wikipedia).
- **Responsive UI**: A conversational UI with bubble-style chat format for an interactive experience.

## Prerequisites

Ensure you have the following prerequisites installed:

- Python 3.x
- Streamlit
- Langchain
- LangGraph
- Groq API (for the chatbot model)
- Arxiv and Wikipedia APIs

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/langgraph_simple_chatbot.git
cd langgraph_simple_chatbot
```


3. Create Python Virtual Environment using below command (The recommended python version is 3.11.0).
    ```bash
    python -m venv venv
                OR
    conda activate -p venv python==3.11.6
    ```

4. Activate Virtual Environment

    ```bash
    .venv/bin/activate 
            OR
    .\venv\Scripts\activate
            OR
    source ./venv/bin/activate
    ```

- if you have used conda to create the virtual environment, use the following command to activate the virtual environment.

    ```bash
    conda activate venv
    ```

5. Install dependencies using below command
    ```bash
    pip install -r requirements.txt
    ```

6. Set up environment variables in a `.env` file.

    - `GROQ_API_KEY`: "please generate put your own api key"
    - `LANGCHAIN_API_KEY` : "please generate put your own api key"

7. To run the `app.py` After installing the dependencies and setting up the API keys, run the Streamlit app:

    ```bash
    streamlit run app.py
    ```


