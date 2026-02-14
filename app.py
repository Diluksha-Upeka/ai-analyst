import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
try:
    from langchain_classic.agents.agent_types import AgentType
except ModuleNotFoundError:  # Back-compat with older LangChain layouts
    from langchain.agents.agent_types import AgentType


@st.cache_resource
def create_agent(_df, api_key, file_name, model_name):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0
    )
    return create_pandas_dataframe_agent(
        llm,
        _df,
        verbose=False,
        allow_dangerous_code=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=15,
    )


# 1. Config & Setup
st.set_page_config(page_title=" AI Data Analyst", layout="wide")
load_dotenv()

# Sidebar for API Key (Optional: allows users to use their own key)
with st.sidebar:
    st.title(" AI Analyst")
    st.markdown("Upload a CSV file and ask questions about your data.")
    
    # Check for env key, otherwise ask user
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter Groq API Key:", type="password")

    model_name = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ], help="Switch to a smaller model if you hit rate limits.")

# 2. Main UI
st.title(" Chat with your Data (CSV)")

# File Uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None and api_key:
    # Detect file change and clear chat history
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []

    # Load Data (cached)
    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)

    df = load_csv(uploaded_file)
    
    # Show Data Preview
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Quick Insights
    st.write("### Quick Insights")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Dataset Summary"):
            st.write(df.describe())

    with col2:
        if st.button("Correlation Matrix"):
            st.write(df.corr(numeric_only=True))

    # 3. Setup the Agent (cached per file)
    agent = create_agent(df, api_key, uploaded_file.name, model_name)

    # 4. Chat Interface
    st.write("### Ask a Question")
    user_question = st.text_input("Example: 'Plot a bar chart of Sales by Region and save it as plot.png'")

    if st.button("Analyze"):
        if user_question:
            with st.spinner("Analyzing data..."):
                try:
                    # A. Cleanup: Delete old plot if it exists
                    if os.path.exists("plot.png"):
                        os.remove("plot.png")

                    # B. The Prompt Injection
                    # We secretly add this instruction to ensure the AI saves the file instead of showing it.
                    enhanced_query = user_question + " If you generate a plot, save it as 'plot.png'. Do not use plt.show()."

                    # C. Run Agent
                    response = agent.run(enhanced_query)
                    st.success("Analysis Complete!")
                    st.write(response)

                    # D. Check for Plot
                    # If the file appeared, display it!
                    if os.path.exists("plot.png"):
                        st.image("plot.png", caption="Generated Visualization")

                except Exception as e:
                    st.error(f"Error: {e}")
                    
elif not api_key:
    st.warning("Please enter an API Key to proceed.")