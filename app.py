import streamlit as st
import pandas as pd
import os
import logging
import importlib
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv


logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ai_data_analyst")

BLOCKED_QUERY_PATTERNS = [
    "__import__",
    "import os",
    "import subprocess",
    "os.system",
    "subprocess",
    "exec(",
    "eval(",
    "open(",
    "rm -rf",
    "del /f",
    "cmd.exe",
    "powershell",
]


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


def is_query_safe(user_question):
    lowered = user_question.lower()
    for pattern in BLOCKED_QUERY_PATTERNS:
        if pattern in lowered:
            return False
    return True


def get_plot_path():
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex

    return artifacts_dir / f"plot_{st.session_state.session_id}.png"


@st.cache_resource
def create_agent(_df, api_key, file_name, model_name):
    try:
        from langchain_groq import ChatGroq
        from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
        try:
            from langchain_classic.agents.agent_types import AgentType
        except ModuleNotFoundError:  # Back-compat with older LangChain layouts
            AgentType = importlib.import_module("langchain.agents.agent_types").AgentType
    except Exception as exc:
        raise RuntimeError(
            "LangChain dependencies are incompatible in this Python environment. "
            "Run the app from the project virtual environment."
        ) from exc

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

    st.markdown("### Safety")
    execution_confirmed = st.checkbox(
        "Enable advanced AI analysis",
        value=False,
        help="This runs AI-generated Python code for data analysis. Use only trusted CSV files.",
    )

# 2. Main UI
st.title(" Chat with your Data (CSV)")

# File Uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None and api_key:
    # Detect file change and clear chat history
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []
        st.session_state.pop("session_id", None)

    df = load_csv(uploaded_file)
    plot_path = get_plot_path()
    
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

    if not execution_confirmed:
        st.info("Enable advanced AI analysis in the sidebar to run custom questions and chart generation.")

    # 4. Chat Interface
    st.write("### Ask a Question")
    user_question = st.text_input("Example: 'Plot a bar chart of Sales by Region and save it as plot.png'")

    if st.button("Analyze"):
        if not execution_confirmed:
            st.warning("Enable advanced AI analysis in the sidebar before running a question.")
        elif not user_question.strip():
            st.warning("Please enter a question first.")
        elif not is_query_safe(user_question):
            st.error("This question looks unsafe. Please remove system or file commands and try again.")
        else:
            with st.spinner("Analyzing data..."):
                try:
                    # Setup the agent only when analysis is explicitly enabled.
                    agent = create_agent(df, api_key, uploaded_file.name, model_name)

                    if plot_path.exists():
                        plot_path.unlink()

                    safe_plot_path = plot_path.as_posix()

                    enhanced_query = (
                        f"{user_question} "
                        f"If you generate a plot, save it as '{safe_plot_path}'. "
                        "Do not use plt.show()."
                    )

                    logger.info(
                        "Analyze request | file=%s | model=%s | question=%s",
                        uploaded_file.name,
                        model_name,
                        user_question[:200],
                    )

                    response = agent.run(enhanced_query)
                    st.success("Analysis Complete!")
                    st.write(response)

                    if plot_path.exists():
                        st.image(str(plot_path), caption="Generated Visualization")

                except Exception as e:
                    logger.exception(
                        "Analyze failed | file=%s | model=%s",
                        uploaded_file.name,
                        model_name,
                    )
                    if "LangChain dependencies are incompatible" in str(e):
                        st.error("Environment issue detected. Start Streamlit with the project venv interpreter.")
                        st.code(".\\.venv\\Scripts\\python.exe -m streamlit run app.py")
                    else:
                        st.error("Analysis failed. Try a simpler question or switch to a smaller model.")
                    with st.expander("Technical details"):
                        st.write(str(e))
                    
elif not api_key:
    st.warning("Please enter an API Key to proceed.")