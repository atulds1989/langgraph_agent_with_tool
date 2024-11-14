# Import necessary libraries
import streamlit as st
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv

# Initialize Streamlit app with a title and introduction
st.set_page_config(page_title="Langchain Chatbot", page_icon="💬", layout="centered")
# st.title("💬 Langchain Tool-Based Chatbot")

# Streamlit interface
st.markdown(
    "<h3 style='text-align: center; color: #333;'>💬 Langchain Tool-Based Chatbot</h3>",
    unsafe_allow_html=True,
)


st.write("Ask a question below, and get responses from either an AI assistant or specific research tools.")

# Load environment variables from the .env file
load_dotenv()

# Set up API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the TypedDict for chatbot state management
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
tools = [wiki_tool, arxiv_tool]  

# Set up the LLM and bind tools
llm = ChatGroq(model_name="Gemma2-9b-It", groq_api_key=groq_api_key)
llm_with_tools = llm.bind_tools(tools=tools)

# Define the chatbot function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Initialize session state for message tracking
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar information
with st.sidebar:
    st.subheader("🤖 Chatbot Information")
    st.write("This chatbot provides intelligent responses by either using an AI assistant or fetching data from research tools such as Arxiv and Wikipedia.")
    st.write("Sources used will be shown next to responses, enabling a clear understanding of where each answer originates from.")

# Input section with an input field and an arrow button in the same row
with st.form(key="input_form", clear_on_submit=True):
    st.markdown("#### Type your query:")
    cols = st.columns([10, 1])  # Two columns for input and send button
    user_input = cols[0].text_input("Your Message", label_visibility="collapsed", placeholder="Ask me anything...")
    submit_button = cols[1].form_submit_button("➤")  # Arrow button



# Display chat history with bubble-style UI
def display_chat():
    # Display conversation history in a more interactive chat format
    st.markdown("###### Chat History")
    for message in st.session_state["messages"]:
        role = message["type"]
        content = message["content"]

        if role == "user":
            # Display user message in a speech bubble style
            st.markdown(
                f'<div style="text-align:right;"><span style="display:inline-block; background-color:#DCF8C6; padding:10px; border-radius:10px;">'
                f'<strong>You:</strong> {content}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            # Display assistant or tool response with source labeling
            role_label = "Assistant" if role == "assistant" else role.capitalize()
            st.markdown(
                f'<div style="text-align:left;"><span style="display:inline-block; background-color:#E8E8E8; padding:10px; border-radius:10px;">'
                f'<strong>{role_label}:</strong> {content}</span></div>',
                unsafe_allow_html=True,
            )



# Process the user input if submitted
if submit_button and user_input.strip():
    # Track user input message
    st.session_state["messages"].append({"type": "user", "content": user_input})
    events = graph.stream({"messages": st.session_state["messages"]}, stream_mode="values")
    
    # Gather responses
    response_text = ""
    source = "Assistant"  # Default source label

    for event in events:
        # Determine the source of the message (tool or assistant)
        if hasattr(event["messages"][-1], "tool_name"):
            source = event["messages"][-1].tool_name.capitalize()  # Tool source
        else:
            source = "Assistant"  # Assistant source

        response = event["messages"][-1].content
        response_text += response + "\n"  # Append response

    # Display the latest response with source label
    st.session_state["messages"].append({"type": source.lower(), "content": response_text.strip()})

display_chat()

# Footer section
def footer():
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot is built using Langsmith and LangGraph. Enjoy interactive conversations!")
    # st.markdown("Developed by **Atul Purohit**")

# Display footer
footer()
