import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import json
import os
import time
import tempfile
import re


# Import custom modules
from utils import (
    read_file, get_data_profile, clean_data, normalize_data,
    get_automatic_visualizations, create_report, 
    process_natural_language_query, get_automatic_insights,
    plot_time_series, plot_bar_chart, fix_dataframe_for_arrow
)
from gemini_agent import GeminiAgent

# Function to process messages and format code blocks
def process_message(content):
    # Process headings
    content = re.sub(r'(\d+)\.\s+\*\*([^*]+)\*\*', r'<h4 style="margin-top:15px;color:#4cc9f0;">\1. \2</h4>', content)
    
    # Function to format code with syntax highlighting
    def format_code(match):
        code = match.group(1).strip()
        # Python keywords
        code = re.sub(r'\b(import|def|return|for|if|else|elif|in|as|class|from|with|try|except|not|and|or|True|False|None)\b', r'<span class="code-keyword">\1</span>', code)
        
        # Python functions
        code = re.sub(r'(print|len|range|str|int|float|list|dict|set|tuple)\(', r'<span class="code-function">\1</span>(', code)
        
        # Pandas functions with stronger highlighting
        pandas_patterns = [
            'groupby', 'mean', 'sum', 'count', 'describe',
            'head', 'tail', 'nlargest', 'nsmallest', 'sort_values',
            'value_counts', 'merge', 'join', 'concat', 'pivot',
            'pivot_table', 'crosstab', 'corr', 'plot'
        ]
        
        for pattern in pandas_patterns:
            code = re.sub(r'\.(' + pattern + r')\(', r'.<span class="pandas-function">\1</span>(', code)
        
        # Comments
        code = re.sub(r'(#.*?)$', r'<span class="code-comment">\1</span>', code, flags=re.MULTILINE)
        
        # Numbers
        code = re.sub(r'\b(\d+\.?\d*)\b', r'<span class="code-number">\1</span>', code)
        
        # Improved string handling - Use a more robust method to handle strings with special chars
        def replace_string(match):
            quote = match.group(1)
            content = match.group(2)
            return f'<span class="code-string">{quote}{content}{quote}</span>'
        
        # Handle single quotes - process one at a time to avoid issues with escaped quotes
        code = re.sub(r"(')((?:(?:\\')|[^'])*?)(')", replace_string, code)
        
        # Handle double quotes - also process one at a time
        code = re.sub(r'(")((?:(?:\\")|[^"])*?)(")', replace_string, code)
        
        return f'<div class="code-block">{code}</div>'
    
    # Function to format results with special styling
    def format_results(match):
        result = match.group(1).strip()
        return f'<div class="results-block">{result}</div>'
    
    # Fix for already formatted code that contains HTML-like tags
    # This will replace "code-keyword", "code-string", etc. with actual spans
    content = re.sub(r'"code-keyword">', r'<span class="code-keyword">', content)
    content = re.sub(r'"code-string">', r'<span class="code-string">', content)
    content = re.sub(r'"code-number">', r'<span class="code-number">', content)
    content = re.sub(r'"code-comment">', r'<span class="code-comment">', content)
    content = re.sub(r'"code-function">', r'<span class="code-function">', content)
    content = re.sub(r'"pandas-function">', r'<span class="pandas-function">', content)
    
    # Detect code sections in analytical responses
    # For example "2. Python/pandas Code:" followed by code
    analysis_code_pattern = r'(\d+\.\s+(?:Python|pandas)(?:/(?:Python|pandas))?\s+Code:.*?)(?=\d+\.\s+|$)'
    
    def format_analysis_section(match):
        section = match.group(1)
        # If there's already code formatting, don't modify
        if '<div class="code-block">' in section:
            return section
            
        # Try to extract code portion and wrap in code block
        # Look for common code patterns
        code_pattern = r'(import.*?|df\s*=.*?|top_\d+.*?=.*?|.*?\.groupby\(.*?\).*?)'
        
        def wrap_code(m):
            code = m.group(1)
            return f'<div class="code-block">{code}</div>'
            
        section = re.sub(code_pattern, wrap_code, section, flags=re.DOTALL)
        return section
    
    content = re.sub(analysis_code_pattern, format_analysis_section, content, flags=re.DOTALL)
    
    # Find any direct code examples not wrapped in code blocks but with patterns like:
    # import pandas as pd
    # Look for common patterns often seen in code examples
    code_patterns = [
        r'(import\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?)',  # import statements
        r'((?:from\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import\s+[a-zA-Z_][a-zA-Z0-9_]*(?:,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)|(?:from\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import\s+\([a-zA-Z0-9_,\s\n]*\)))',  # from ... import
        r'(def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*:)',  # function definitions
        r'(class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*\([^)]*\))?\s*:)',  # class definitions
        r'((?:if|for|while|with)\s+[^:]+:)',  # control flow statements
    ]
    
    # Wrap sections that look like code in code blocks if they're not already in one
    for pattern in code_patterns:
        def replace_with_code_block(match):
            code = match.group(1)
            # Check if this code is already inside a code-block div
            if '<div class="code-block">' in content[:match.start()] and '</div>' in content[match.end():]:
                # Already in a code block, return as is
                return code
            return f'<code class="inline-code">{code}</code>'
        
        content = re.sub(pattern, replace_with_code_block, content)
    
    # Handle code blocks with ```python
    if '```python' in content:
        content = re.sub(r'```python(.*?)```', '', content, flags=re.DOTALL)
    
    # Handle other ```code blocks
    if '```' in content:
        content = re.sub(r'```(.*?)```', '', content, flags=re.DOTALL)
    
    # Find and format any output/result sections
    # Use a more comprehensive pattern to match various output formats
    output_pattern = r'(?:Output:|Result:|Example output:|This would output:|The output would be:|The result is:|Returns:|Resulting dataframe:|Prints:|Example result:)(.*?)(?=\n\n|\n[A-Z#]|$)'
    content = re.sub(output_pattern, 
                   r'<strong style="color:#66c75a;display:block;margin-top:12px;margin-bottom:6px;">Output:</strong><div class="results-block">\1</div>', 
                   content, flags=re.DOTALL)
    
    # Format "Answer:" sections
    answer_pattern = r'(\n|^)(\d+\.\s+)?(?:Answer:|Conclusion:|Final answer:|In conclusion:|To summarize:|Key finding:|Summary:)(.*?)(?=\n\n|\n[A-Z#]|$)'
    content = re.sub(answer_pattern, 
                   r'\1<div class="answer-block"><strong style="color:#4cc9f0;font-size:17px;display:block;margin-bottom:6px;">Answer:</strong>\3</div>', 
                   content, flags=re.DOTALL)
    
    # Add code blocks for 2-3 line code examples that might not be caught otherwise
    code_example_pattern = r'(\n|^)(df\.[a-zA-Z_]+\([^)]*\)(?:\.[a-zA-Z_]+\([^)]*\))*)'
    content = re.sub(code_example_pattern, r'\1<code class="inline-code">\2</code>', content)
    
    # Handle other formatting
    content = content.replace('\n', '<br>')
    
    # Remove predefined structure for direct answers
    # Allow model to respond directly without code unless necessary
    content = re.sub(r'\b(\d+\.\s+\*\*[^*]+\*\*)\b', '', content)
    
    return content

# Initialize Gemini agent before we show the UI
try:
    agent = GeminiAgent()
    if not hasattr(agent, 'initialized') or not agent.initialized:
        api_key_error = True
    else:
        api_key_error = False
except Exception as e:
    agent = None
    api_key_error = True
    print(f"Error initializing Gemini agent: {str(e)}")

# Set page configuration
st.set_page_config(
    page_title="Insight Genie",
    page_icon="🧞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_profile' not in st.session_state:
    st.session_state.data_profile = None
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'agent' not in st.session_state:
    st.session_state.agent = agent
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key_error' not in st.session_state:
    st.session_state.api_key_error = api_key_error

# Custom CSS
st.markdown("""
<style>
    /* Global styles */
    .main {
        padding: 1rem;
        max-width: 100%;
    }
    
    /* Typography */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #333;
        border-left: 4px solid #1E88E5;
        padding-left: 10px;
    }
    
    /* Card styles */
    .card {
        border-radius: 8px;
        border: 1px solid #e6e6e6;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    .insight-card {
        background-color: #f8f9fa;
        border-left: 4px solid #4285f4;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Chat message styling */
    .chat-msg {
        padding: 15px;
        margin: 10px 0;
        border-radius: 15px;
        max-width: 85%;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        overflow-wrap: break-word;
        white-space: pre-wrap;
    }
    
    .user-msg {
        background-color: #1E88E5;
        color: white;
        border-radius: 18px 18px 0 18px;
        float: right;
        clear: both;
    }
    
    .bot-msg {
        background-color: #f8f9fa;
        color: #333;
        border-radius: 18px 18px 18px 0;
        float: left;
        clear: both;
        font-size: 16px;
        line-height: 1.6;
        padding: 18px;
    }
    
    /* Code block styling */
    .code-block {
        background-color: #f5f5f5;
        color: #333;
        padding: 18px;
        border-radius: 8px;
        overflow-x: auto;
        margin: 18px 0;
        font-size: 16px;
        line-height: 1.6;
        white-space: pre;
        border-left: 4px solid #007acc;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    /* Results block styling */
    .results-block {
        font-family: 'Consolas', 'Courier New', monospace;
        background-color: #0f2318;
        color: #a8e890;
        padding: 16px 18px;
        border-radius: 8px;
        margin: 16px 0;
        font-size: 16px;
        line-height: 1.5;
        border-left: 4px solid #66c75a;
        white-space: pre;
        overflow-x: auto;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
    }
    
    /* Light mode syntax highlighting */
    .code-comment { color: #608b4e; font-style: italic; }
    .code-keyword { color: #c586c0; font-weight: bold; }
    .code-string { color: #ce9178; }
    .code-number { color: #b5cea8; }
    .code-function { color: #4fc1ff; }
    .code-class { color: #4ec9b0; }
    .code-variable { color: #9cdcfe; }
    
    /* Enhanced pandas code styling */
    .pandas-function { color: #ffca28; font-weight: bold; }
    
    /* Code block improvements */
    .bot-msg .code-block {
        max-width: 100%;
        margin: 12px 0;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Results block improvements */
    .bot-msg .results-block {
        max-width: 100%;
        margin: 12px 0;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        color: #addc91;
    }
    
    /* Fixed code-formatting within text */
    span.code-keyword, span.code-string, span.code-number, 
    span.code-comment, span.code-function, span.pandas-function {
        font-family: 'Consolas', 'Courier New', monospace;
        display: inline-block;
        padding: 0 2px;
    }
    
    /* Inline code style for better readability */
    code, .inline-code {
        font-family: 'Consolas', 'Courier New', monospace;
        background-color: #f0f0f0;
        color: #0e1e25;
        padding: 2px 4px;
        border-radius: 3px;
        font-size: 0.9em;
        border: 1px solid #e0e0e0;
    }
    
    /* Answer block styling */
    .answer-block {
        background-color: #e0f7fa; /* Light cyan background for visibility */
        border-left: 4px solid #00796b; /* Teal border for emphasis */
        padding: 16px 18px;
        margin: 18px 0;
        border-radius: 8px;
        font-weight: 500;
        font-size: 16px;
        line-height: 1.6;
        color: #004d40; /* Dark teal text for contrast */
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .analysis-section {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    
    .code-section {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    .explanation-section {
        background-color: #f3e5f5;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #9C27B0;
    }
    
    .answer-section {
        background-color: #ffebee;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #F44336;
    }

    
    
    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .bot-msg {
            background-color: #282c34;
            color: #dcdfe4;
        }
        
        .code-block {
        background-color: #263238;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Consolas', monospace;
        white-space: pre-wrap;
        margin: 1rem 0;
    }
    
    .code-block pre {
        margin: 0;
        padding: 0;
    }
        
        .results-block {
            background-color: #0f2318;
            border-color: #66c75a;
            color: #a8e890;
            box-shadow: 0 3px 10px rgba(0,0,0,0.4);
        }
        
        .card {
            background-color: #282c34;
            border-color: #444;
        }
        
        .insight-card {
            background-color: #282c34;
        }
        
        .answer-block {
            background-color: #1a3347;
            color: #c5e8ff;
            border-color: #4cc9f0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        }
    }
    
    /* Button styles */
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
        width: 100%;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1976D2;
    }
    
    /* Chat interface */
    .chat-input {
        border-radius: 20px;
        border: 1px solid #e6e6e6;
        padding: 10px 15px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        padding: 8px 16px;
    }
    
    .msg-container {
        overflow-y: auto;
        height: 350px;
        margin-bottom: 15px;
        padding: 15px;
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Enhanced Chat Interface */
    .nl-chat-container {
        height: 450px;
        background-color: #f9f9f9;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
    }
    
    .nl-chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        display: flex;
        flex-direction: column;
    }
    
    .nl-chat-input-area {
        padding: 15px;
        background-color: white;
        border-top: 1px solid #e0e0e0;
        border-radius: 0 0 12px 12px;
    }
    
    .nl-user-msg {
        background-color: #1E88E5;
        color: white;
        border-radius: 20px 20px 0 20px;
        padding: 12px 18px;
        margin: 8px 0;
        align-self: flex-end;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.3s ease-in-out;
    }
    
    .nl-bot-msg {
        background-color: white;
        color: #333;
        border-radius: 20px 20px 20px 0;
        padding: 12px 18px;
        margin: 8px 0;
        align-self: flex-start;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        animation: fadeIn 0.3s ease-in-out;
        border: 1px solid #f0f0f0;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .nl-chat-button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 30px;
        padding: 10px 20px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        width: 100%;
    }
    
    .nl-chat-button:hover {
        background-color: #1976D2;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .nl-suggestion-chip {
        display: inline-block;
        background-color: #f0f4f8;
        color: #333;
        border-radius: 30px;
        padding: 8px 16px;
        margin: 5px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #e0e6ed;
    }
    
    .nl-suggestion-chip:hover {
        background-color: #e0e6ed;
        transform: scale(1.03);
    }
    
    /* Suggestion buttons */
    .suggestion-btn {
        background-color: #f1f1f1;
        border: none;
        border-radius: 15px;
        padding: 6px 12px;
        margin: 5px;
        cursor: pointer;
        font-size: 0.8rem;
        transition: background-color 0.3s;
    }
    
    .suggestion-btn:hover {
        background-color: #e0e0e0;
    }
    
    /* Responsive layout adjustments */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .section-title {
            font-size: 1.2rem;
        }
        
        .card {
            padding: 1rem;
        }
        
        .msg-container {
            height: 300px;
        }
    }
    
    /* Layout improvements */
    .data-container {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        border-bottom: none !important;
    }
    
    .stTabs [data-baseweb="tab"] {
         height: 60px;  /* Increased height */
    font-size: 1.2rem !important;  /* Increased text size */
    white-space: pre-wrap;
    background-color: #f1f1f1;
    border-radius: 4px 4px 0 0;
    padding: 15px 20px !important;  /* Increased padding */
    border: none !important;  /* Remove all borders */
    }
    .card .stTabs [data-baseweb="tab"] {
    font-size: 1.1rem !important;  /* Slightly smaller but still increased */
    height: 50px;
    padding: 12px 18px !important;
}
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
        box-shadow: none !important;
    }
    
    /* Download link styling */
    .download-link {
        display: inline-block;
        margin-top: 10px;
        padding: 10px 20px;
        background-color: #4CAF50;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    
    .download-link:hover {
        background-color: #3e8e41;
    }
</style>
""", unsafe_allow_html=True)

# Main UI
st.markdown("<h1 class='main-title'>AI Insight Genie </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; margin-bottom: 2rem;'>Your AI-powered data insights assistant</p>", unsafe_allow_html=True)

# Display API key error warning if needed
if st.session_state.api_key_error:
    st.error("⚠️ **Google Gemini API key error:** AI features are limited. Please check your API key in the .env file or make sure you have internet connectivity.")

# Use tabs for better organization with improved names
tabs = st.tabs(["📊 Data Analysis", "💬 Chat with Data", "📝 Generate Report"])

with tabs[0]:
    # Data Analysis Dashboard
    st.markdown("""
    <div style='text-align: center;'>
        <h2>📊 Data Analysis Dashboard</h2>
        <p style='color: #666;'>Upload your dataset to visualize insights and patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Left side - Data Upload and Preview
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("### Upload Dataset")
        with st.container():
            st.markdown('<div class="card" style="padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
            file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
            if file is not None:
                with st.spinner("Processing data..."):
                    st.session_state.df = read_file(file)
                    # Fix DataFrame for Arrow compatibility
                    st.session_state.df = fix_dataframe_for_arrow(st.session_state.df)
                    st.session_state.data_profile = get_data_profile(st.session_state.df)
                    st.success(f"✅ Successfully loaded {file.name}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show dataset preview if available
        if st.session_state.df is not None:
            st.markdown("### Dataset Preview")
            with st.container():
                st.markdown('<div class="card" style="padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
                st.dataframe(st.session_state.df.head(5), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### Data Cleaning Tools")
            with st.container():
                st.markdown('<div class="card" style="padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
                
                cleaning_tab1, cleaning_tab2 = st.tabs(["Missing Values", "Outliers"])
                
                with cleaning_tab1:
                    cleaning_method = st.selectbox(
                        "Handle missing values",
                        ["Drop missing rows", "Fill with mean", "Fill with median", "Fill with mode"]
                    )
                    if st.button("Apply", key="apply_cleaning"):
                        method_map = {
                            "Drop missing rows": "drop",
                            "Fill with mean": "mean",
                            "Fill with median": "median",
                            "Fill with mode": "mode"
                        }
                        with st.spinner("Cleaning data..."):
                            st.session_state.df = clean_data(st.session_state.df, method=method_map[cleaning_method])
                            st.session_state.data_profile = get_data_profile(st.session_state.df)
                            st.success("✅ Data cleaned successfully!")
                
                with cleaning_tab2:
                    num_cols = st.session_state.df.select_dtypes(include=["number"]).columns.tolist()
                    col_for_outliers = st.selectbox("Select column for outlier detection", num_cols)
                    if st.button("Handle Outliers"):
                        # Simple IQR-based outlier handling
                        with st.spinner("Handling outliers..."):
                            Q1 = st.session_state.df[col_for_outliers].quantile(0.25)
                            Q3 = st.session_state.df[col_for_outliers].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            # Create a mask for outliers
                            outlier_mask = (st.session_state.df[col_for_outliers] < lower_bound) | (st.session_state.df[col_for_outliers] > upper_bound)
                            num_outliers = outlier_mask.sum()
                            
                            # Handle outliers by capping
                            st.session_state.df.loc[st.session_state.df[col_for_outliers] < lower_bound, col_for_outliers] = lower_bound
                            st.session_state.df.loc[st.session_state.df[col_for_outliers] > upper_bound, col_for_outliers] = upper_bound
                            
                            st.success(f"✅ Handled {num_outliers} outliers in {col_for_outliers}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
    # Right side - Data Profile, Visualizations and Insights
    with col2:
        if st.session_state.df is not None:
            # Dashboard Overview
            st.markdown("### Dashboard Overview")
            
            # KPI metrics
            metric1, metric2, metric3, metric4 = st.columns(4)
            
            with metric1:
                st.metric(
                    label="Rows", 
                    value=f"{st.session_state.data_profile['rows']:,}"
                )
            
            with metric2:
                st.metric(
                    label="Columns", 
                    value=st.session_state.data_profile['columns']
                )
            
            with metric3:
                st.metric(
                    label="Missing Values", 
                    value=f"{st.session_state.data_profile['missing_values_pct']}%"
                )
            
            with metric4:
                num_cols = len(st.session_state.data_profile['numerical_columns'])
                cat_cols = len(st.session_state.data_profile['categorical_columns'])
                st.metric(
                    label="Column Types", 
                    value=f"{num_cols} num, {cat_cols} cat"
                )
            
            # Tabs for different analysis sections
            analysis_tabs = st.tabs(["📋 Data Profile", "📊 Visualizations", "🧠 Insights"])
            
            with analysis_tabs[0]:
                st.markdown("#### Dataset Summary")
                
                # Create a more detailed profile
                col_profile1, col_profile2 = st.columns(2)
                
                with col_profile1:
                    st.markdown("##### Column Information")
                    column_info = pd.DataFrame({
                        'Column': st.session_state.df.columns,
                        'Type': st.session_state.df.dtypes,
                        'Missing': st.session_state.df.isna().sum(),
                        'Missing %': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2)
                    })
                    st.dataframe(column_info, use_container_width=True)
                
                with col_profile2:
                    st.markdown("##### Numerical Statistics")
                    st.dataframe(st.session_state.df.describe().round(2), use_container_width=True)
                
            with analysis_tabs[1]:
                st.markdown("#### Visualizations")
                
                if 'visualizations' not in st.session_state or not st.session_state.visualizations:
                    with st.spinner("Generating visualizations..."):
                        st.session_state.visualizations = get_automatic_visualizations(st.session_state.df)
                
                # Filter visualizations by type
                viz_types = ["All"] + list(set(["Histogram", "Scatter", "Bar", "Box", "Line"]))
                selected_type = st.selectbox("Filter visualizations by type:", viz_types)
                
                # Display visualizations
                viz_count = 0
                if st.session_state.visualizations:
                    for viz in st.session_state.visualizations:
                        viz_type = viz.data[0].type if hasattr(viz, "data") and len(viz.data) > 0 else "Other"
                        if selected_type == "All" or viz_type.capitalize() == selected_type:
                            st.plotly_chart(viz, use_container_width=True)
                            viz_count += 1
                    
                    if viz_count == 0:
                        st.info(f"No visualizations of type '{selected_type}' found.")
                else:
                    st.info("No visualizations available. Please check your dataset.")
                
                # Custom visualization options
                st.markdown("#### Create Custom Visualization")
                custom_viz_type = st.selectbox(
                    "Select visualization type",
                    ["Scatter Plot", "Bar Chart", "Histogram", "Box Plot", "Line Chart"]
                )
                
                if custom_viz_type == "Scatter Plot":
                    num_cols = st.session_state.df.select_dtypes(include=["number"]).columns.tolist()
                    x_col = st.selectbox("X-axis", num_cols, key="scatter_x")
                    y_col = st.selectbox("Y-axis", num_cols, key="scatter_y")
                    
                    if st.button("Create Scatter Plot"):
                        fig = px.scatter(st.session_state.df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif custom_viz_type == "Bar Chart":
                    cat_cols = st.session_state.df.select_dtypes(include=["object", "category"]).columns.tolist()
                    if cat_cols:
                        x_col = st.selectbox("Category", cat_cols)
                        num_cols = st.session_state.df.select_dtypes(include=["number"]).columns.tolist()
                        
                        if num_cols:
                            y_col = st.selectbox("Value (optional)", ["Count"] + num_cols)
                            
                            if st.button("Create Bar Chart"):
                                if y_col == "Count":
                                    fig = plot_bar_chart(st.session_state.df, x_col)
                                else:
                                    fig = plot_bar_chart(st.session_state.df, x_col, y_col)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            if st.button("Create Bar Chart"):
                                fig = plot_bar_chart(st.session_state.df, x_col)
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No categorical columns found for bar chart.")
                
                elif custom_viz_type == "Histogram":
                    num_cols = st.session_state.df.select_dtypes(include=["number"]).columns.tolist()
                    if num_cols:
                        col = st.selectbox("Column", num_cols)
                        bins = st.slider("Number of bins", 5, 50, 20)
                        
                        if st.button("Create Histogram"):
                            fig = px.histogram(st.session_state.df, x=col, nbins=bins, title=f"Distribution of {col}")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No numerical columns found for histogram.")
            
            with analysis_tabs[2]:
                st.markdown("#### AI-Generated Insights")
                
                if st.button("Generate Insights", key="generate_insights_btn"):
                    with st.spinner("Analyzing data and generating insights..."):
                        try:
                            if st.session_state.agent is None:
                                st.error("The AI assistant is not available. Please check your API key in the .env file.")
                            else:
                                st.session_state.insights = st.session_state.agent.generate_insights(
                                    st.session_state.df, 
                                    st.session_state.visualizations, 
                                    st.session_state.data_profile
                                )
                        except Exception as e:
                            st.error(f"Error generating insights: {str(e)}")
                
                # Display insights
                if st.session_state.insights:
                    for i, insight in enumerate(st.session_state.insights):
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; 
                                    border-left: 4px solid #1E88E5; 
                                    padding: 15px; 
                                    margin-bottom: 10px; 
                                    border-radius: 4px;">
                            <p>📊 {insight}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Generate Data Summary
                st.markdown("#### AI Data Summary")
                if st.button("Generate Summary", key="generate_summary_btn"):
                    with st.spinner("Generating data summary..."):
                        summary = st.session_state.agent.generate_data_summary(
                            st.session_state.data_profile,
                            st.session_state.df
                        )
                        st.markdown(f"""
                        <div style="background-color: #e8f4fd; 
                                    border: 1px solid #1E88E5; 
                                    padding: 15px; 
                                    margin-top: 10px;
                                    margin-bottom: 10px; 
                                    border-radius: 4px;">
                            <p>{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Please upload a dataset to begin exploration.")
            
            # Show sample dataset options when no data is uploaded
            st.markdown("### Try a sample dataset")
            sample_datasets = ["Iris Flower Dataset", "Titanic Passenger Data", "Housing Prices"]
            selected_sample = st.selectbox("Select a sample dataset to explore:", sample_datasets)
            
            if st.button("Load Sample Data"):
                with st.spinner(f"Loading {selected_sample}..."):
                    if selected_sample == "Iris Flower Dataset":
                        from sklearn.datasets import load_iris
                        data = load_iris()
                        df = pd.DataFrame(data.data, columns=data.feature_names)
                        df['species'] = pd.Categorical.from_codes(data.target, data.target_names)
                        st.session_state.df = fix_dataframe_for_arrow(df)
                    
                    elif selected_sample == "Titanic Passenger Data":
                        titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                        df = pd.read_csv(titanic_url)
                        st.session_state.df = fix_dataframe_for_arrow(df)
                    
                    elif selected_sample == "Housing Prices":
                        from sklearn.datasets import fetch_california_housing
                        housing = fetch_california_housing()
                        df = pd.DataFrame(
                            housing.data, 
                            columns=housing.feature_names
                        )
                        df['PRICE'] = housing.target
                        st.session_state.df = fix_dataframe_for_arrow(df)
                    
                    # Generate data profile for the sample data
                    st.session_state.data_profile = get_data_profile(st.session_state.df)
                    st.success(f"✅ Successfully loaded {selected_sample}")
                    st.rerun()

with tabs[1]:
    # Chat with Data using Gemini API
    st.markdown("""
    <div style='text-align: center;'>
        <h2>💬 Chat with Your Data using Gemini API</h2>
        <p style='color: #666;'>Ask questions about your dataset in natural language</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a chat interface
    with st.container():
        st.markdown('<div class="card" style="padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
        
        try:
            if st.session_state.df is not None:
                # Initialize chat history if not exists
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                # Chat container
                chat_container = st.container()
                with chat_container:
                    # Display chat messages with better styling
                    if st.session_state.chat_history:
                        # Create a clear message about checking chat history
                        st.success("✅ Question processed! Scroll down to see the answer in the chat messages.")
                        
                        st.markdown("""
                        <div style='height: 500px; overflow-y: auto; padding: 15px; 
                                  border: 1px solid #e0e0e0; border-radius: 10px; 
                                  background-color: #fafafa; margin-bottom: 20px;'
                             id='chat-messages-container'>
                        <a name="chat-messages"></a>
                        """, unsafe_allow_html=True)
                        
                        for i, msg in enumerate(st.session_state.chat_history):
                            # Format the message content with the code_formatter function
                            formatted_content = process_message(msg['content'])
                            
                            # Apply appropriate styling based on message role
                            if msg['role'] == 'user':
                                st.markdown(f"""
                                <div class='chat-msg user-msg'>
                                    <div class='msg-content'>{formatted_content}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:  # ai or assistant messages
                                # For bot messages, we need to be extra careful with HTML
                                st.markdown(f"""
                                <div class='chat-msg bot-msg'>
                                    <div class='msg-content'>{formatted_content}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Add JavaScript for auto-scrolling to the bottom
                        st.markdown("""
                        <script>
                            function scrollChatToBottom() {
                                var chatContainer = document.getElementById('chat-messages-container');
                                if (chatContainer) {
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                }
                            }
                            
                            // Scroll when content is loaded
                            window.onload = function() {
                                setTimeout(scrollChatToBottom, 500);
                            }
                            
                            // Additional scroll trigger for when content updates
                            const observer = new MutationObserver(scrollChatToBottom);
                            const config = { childList: true, subtree: true };
                            
                            setTimeout(function() {
                                var chatContainer = document.getElementById('chat-messages-container');
                                if (chatContainer) {
                                    observer.observe(chatContainer, config);
                                    scrollChatToBottom();
                                }
                            }, 1000);
                        </script>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)  # Close the chat-messages div
                    else:
                        st.info("Ask a question about your dataset to get started!")
                
                # Chat input
                user_input = st.chat_input("Ask a question about your data...")
                
                # Divider
                st.markdown("<hr style='margin-top: 1rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)
                
                # # Display suggested questions if chat is empty
                # if len(st.session_state.chat_history) == 0:
                #     # Remove the problematic suggested questions section
                #     st.info("Ask a question about your dataset in the chat input above to analyze your data.")
                
                # Process user input if provided
                if user_input:
                    # Add user message to chat
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
                    # Process with Gemini
                    with st.spinner("Analyzing data..."):
                        # Prepare context
                        numerical_cols = st.session_state.data_profile['numerical_columns'][:5]  
                        categorical_cols = st.session_state.data_profile['categorical_columns'][:5]  
                        
                        # Statistics for key numerical columns
                        stats_summary = ""
                        for col in numerical_cols:
                            try:
                                # Check if column is numeric before calculating stats
                                if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                                    stats_summary += (f"{col}: min={st.session_state.df[col].min():.2f}, "
                                                    f"max={st.session_state.df[col].max():.2f}, "
                                                    f"mean={st.session_state.df[col].mean():.2f}\n")
                                else:
                                    stats_summary += f"{col}: non-numeric column\n"
                            except Exception as stats_error:
                                stats_summary += f"{col}: error calculating stats\n"
                                st.session_state.error_message = f"Error with statistics for {col}: {str(stats_error)}"
                        
                        # Enhanced sample data with more rows
                        sample_data = st.session_state.df.head(5).to_string()
                        
                        # List all available columns
                        all_columns = ", ".join(st.session_state.df.columns.tolist())
                        
                        # Add column data types
                        column_types = {}
                        for col in st.session_state.df.columns:
                            column_types[col] = str(st.session_state.df[col].dtype)
                        
                        # Generate code for data summary
                        data_summary_code = f"""
                        # Data summary
                        df.shape  # ({st.session_state.data_profile['rows']}, {st.session_state.data_profile['columns']})
                        
                        # Column data types
                        df.dtypes
                        
                        # First 5 rows
                        df.head()
                        
                        # Summary statistics
                        df.describe()
                        """
                        
                        # Add some common analyses that might be helpful
                        if len(numerical_cols) >= 2:
                            corr_analysis = f"""
                            # Correlation between numerical columns
                            correlation_matrix = df[{numerical_cols}].corr()
                            correlation_matrix
                            """
                        else:
                            corr_analysis = "# Not enough numerical columns for correlation analysis"
                        
                        # Add value counts for categorical columns
                        cat_value_counts = ""
                        for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
                            try:
                                # Check if the column exists and has fewer than 100 unique values
                                if col in st.session_state.df.columns and st.session_state.df[col].nunique() < 100:
                                    counts = st.session_state.df[col].value_counts().head(5)
                                    counts_str = "\n".join([f"# {val}: {count}" for val, count in counts.items()])
                                    cat_value_counts += f"""
                                    # Top values for {col}:
                                    {counts_str}
                                    
                                    """
                                else:
                                    cat_value_counts += f"# {col}: too many unique values or column not found\n"
                            except Exception as cat_error:
                                cat_value_counts += f"# Error analyzing {col}: {str(cat_error)}\n"
                                st.session_state.error_message = f"Error with categorical values for {col}: {str(cat_error)}"
                        
                        # Full prompt
                        full_prompt = f"""
                        You are a data analysis assistant. I have a dataset with the following details:
                        
                        Dataset has {st.session_state.data_profile['rows']} rows and {st.session_state.data_profile['columns']} columns.
                        
                        All available columns: {all_columns}
                        
                        Column data types:
                        {json.dumps(column_types, indent=2)}
                        
                        Numerical columns: {', '.join(numerical_cols)}
                        Categorical columns: {', '.join(categorical_cols)}
                        
                        Statistics:
                        {stats_summary}
                        
                        Sample data (first 5 rows):
                        {sample_data}
                        
                        Data summary code:
                        {data_summary_code}
                        
                        {corr_analysis}
                        
                        Categorical column value counts:
                        {cat_value_counts}
                        
                        Previous conversation:
                        {' '.join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_history[:-1]])}
                        
                        User question: {user_input}
                        
                        Answer with specific information from the dataset using Python/pandas code. Follow these steps:
                        1. First, understand what analysis is needed based on the question
                        2. Show the Python/pandas code that would perform this analysis (using actual column names from this dataset)
                        3. Explain what the code does in a simple way
                        4. Provide a direct answer based on the data
                        
                        For any grouping operations, do NOT list every single group - focus on the top 5-10 results at most.
                        Keep the response concise and focused on answering the specific question.
                        """
                        
                        # Get answer from Gemini
                        try:
                            if st.session_state.agent is None:
                                response = "The AI assistant is not available. Please check your API key in the .env file."
                            else:
                                response = st.session_state.agent.model.generate_content(full_prompt).text
                        except Exception as e:
                            response = f"I encountered an error analyzing your data: {str(e)}. Please try a different question or check if your dataset is properly formatted."
                        
                        # Add response to chat
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # Force rerun to show the new messages
                        st.rerun()
                
                # Option to reset chat
                if st.session_state.chat_history:
                    if st.button("Clear Chat History", key="clear_chat"):
                        st.session_state.chat_history = []
                        st.rerun()
        except Exception as e:
            st.error(f"An error occurred in the chat interface: {str(e)}")
            st.info("Please try refreshing the page or upload your dataset again.")
        
        st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    # Report Generation with Gemini
    st.markdown("""
    <div style='text-align: center;'>
        <h2>📝 Generate Interactive Report</h2>
        <p style='color: #666;'>Create comprehensive data reports with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        # Report configuration section
        with st.container():
            st.markdown('<div class="card" style="padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
            
            st.markdown("### Report Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_title = st.text_input("Report Title", value="Data Analysis Report")
                include_summary = st.checkbox("Include Executive Summary", value=True)
                include_visualizations = st.checkbox("Include Visualizations", value=True)
            
            with col2:
                include_insights = st.checkbox("Include AI Insights", value=True)
                include_recommendations = st.checkbox("Include AI Recommendations", value=True)
                report_format = st.selectbox("Report Format", ["HTML", "PDF (Coming soon)"])
            
            # Generate Report button
            if st.button("Generate Report", key="generate_report_btn"):
                with st.spinner("Generating comprehensive report..."):
                    # If insights are not already generated, generate them
                    if include_insights and (not st.session_state.insights or len(st.session_state.insights) == 0):
                        st.session_state.insights = st.session_state.agent.generate_insights(
                            st.session_state.df, 
                            st.session_state.visualizations, 
                            st.session_state.data_profile
                        )
                    
                    # Generate executive summary with Gemini if needed
                    executive_summary = ""
                    if include_summary:
                        try:
                            executive_summary = st.session_state.agent.generate_data_summary(
                                st.session_state.data_profile,
                                st.session_state.df
                            )
                        except Exception as e:
                            st.warning(f"Could not generate summary: {str(e)}")
                            executive_summary = "Executive summary could not be generated."
                    
                    # Generate recommendations with Gemini if needed
                    recommendations = []
                    if include_recommendations:
                        try:
                            # Create prompt for recommendations
                            prompt = f"""
                            Based on the following dataset information, provide 3-5 actionable recommendations:
                            
                            Dataset has {st.session_state.data_profile['rows']} rows and {st.session_state.data_profile['columns']} columns.
                            
                            Numerical columns: {', '.join(st.session_state.data_profile['numerical_columns'][:5])}
                            Categorical columns: {', '.join(st.session_state.data_profile['categorical_columns'][:5])}
                            
                            Key insights:
                            {'. '.join(st.session_state.insights[:3]) if st.session_state.insights else 'No insights available.'}
                            
                            Provide specific, actionable recommendations for further analysis or business decisions.
                            Each recommendation should be concise and data-driven.
                            Format each recommendation as a separate point.
                            """
                            
                            response = st.session_state.agent.model.generate_content(prompt).text
                        except Exception as e:
                            st.warning(f"Could not generate recommendations: {str(e)}")
                            recommendations = ["No recommendations available."]
                    
                    # Create the report
                    html_report = create_report(
                        st.session_state.df,
                        st.session_state.data_profile,
                        st.session_state.visualizations if include_visualizations else [],
                        st.session_state.insights if include_insights else [],
                        executive_summary=executive_summary if include_summary else "",
                        recommendations=recommendations if include_recommendations else [],
                        title=report_title
                    )
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
                        f.write(html_report)
                        temp_file_path = f.name
                    
                    # Convert to downloadable format
                    with open(temp_file_path, "r", encoding="utf-8") as f:
                        report_html = f.read()
                    
                    # Encode for download
                    b64_html = base64.b64encode(report_html.encode()).decode()
                    
                    # Create download link
                    download_filename = report_title.lower().replace(" ", "_") + ".html"
                    href = f'<a href="data:text/html;base64,{b64_html}" download="{download_filename}" class="download-link">📥 Download Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Clean up temporary file
                    os.unlink(temp_file_path)
                    
                    st.success("✅ Report generated successfully!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Report preview
        st.markdown("### Report Preview")
        with st.expander("Click to see what's included in the report"):
            preview_col1, preview_col2 = st.columns(2)
            
            with preview_col1:
                st.markdown("""
                #### Data Overview
                - Dataset Summary
                - Column Profiles
                - Data Quality Metrics
                
                #### Visualizations
                - Data Distribution Charts
                - Correlation Analysis
                - Key Trends
                """)
            
            with preview_col2:
                st.markdown("""
                #### AI-Generated Analysis
                - Executive Summary
                - Key Insights
                - Anomaly Detection
                
                #### Recommendations
                - Data-Driven Suggestions
                - Further Analysis Ideas
                - Business Implications
                """)
    else:
        st.info("Please upload a dataset in the Data Analysis tab first to generate a report.")

# Display footer
st.markdown('<hr style="margin-top: 2rem;">', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; padding: 1rem 0;">Insight Genie © 2025 | Powered by Gemini API and Streamlit</p>', unsafe_allow_html=True)

# Main function
if __name__ == "__main__":
    pass  # Already running in Streamlit