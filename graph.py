import streamlit as st
import google.generativeai as genai
from reasoning_libs.cot_reasoning import (
    VisualizationConfig,
    create_mermaid_diagram as create_cot_diagram,
    parse_cot_response
)
from reasoning_libs.tot_reasoning import (
    create_mermaid_diagram as create_tot_diagram,
    parse_tot_response
)
from reasoning_libs.l2m_reasoning import (
    create_mermaid_diagram as create_l2m_diagram,
    parse_l2m_response
)
from reasoning_libs.selfconsistency_reasoning import (
    create_mermaid_diagram as create_scr_diagram,
    parse_scr_response
)
from reasoning_libs.selfrefine_reasoning import (
    create_mermaid_diagram as create_srf_diagram,
    parse_selfrefine_response
)
from reasoning_libs.bs_reasoning import (
    create_mermaid_diagram as create_bs_diagram,
    parse_bs_response
)
from reasoning_libs.configs import general as config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Google Generative AI
API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"  # Replace with your actual API key
genai.configure(api_key=API_KEY)

# Streamlit App Title
st.title("ReasonGraph: Visualization of Reasoning Paths")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    reasoning_method = st.selectbox(
        "Reasoning Method",
        options=["cot", "tot", "l2m", "scr", "srf", "bs"],
        index=0
    )
    max_tokens = st.number_input("Max Tokens", value=config.max_tokens, min_value=1)
    chars_per_line = st.number_input("Characters per Line", value=config.chars_per_line, min_value=1)
    max_lines = st.number_input("Max Lines", value=config.max_lines, min_value=1)

# Main Input Section
st.header("Input Your Query")
question = st.text_area("Enter your question or task:")

# Process Button
if st.button("Process"):
    if not question:
        st.error("Question is required!")
    else:
        try:
            # Initialize Google Generative AI model
            model = genai.GenerativeModel('gemini-pro')  # Use the Gemini Pro model

            # Generate Response
            st.info("Generating response...")
            response = model.generate_content(question)
            raw_response = response.text

            # Create Visualization Config
            viz_config = VisualizationConfig(
                max_chars_per_line=chars_per_line,
                max_lines=max_lines
            )

            # Generate Visualization Based on Reasoning Method
            visualization = None
            if reasoning_method == 'cot':
                result = parse_cot_response(raw_response, question)
                visualization = create_cot_diagram(result, viz_config)
            elif reasoning_method == 'tot':
                result = parse_tot_response(raw_response, question)
                visualization = create_tot_diagram(result, viz_config)
            elif reasoning_method == 'l2m':
                result = parse_l2m_response(raw_response, question)
                visualization = create_l2m_diagram(result, viz_config)
            elif reasoning_method == 'scr':
                result = parse_scr_response(raw_response, question)
                visualization = create_scr_diagram(result, viz_config)
            elif reasoning_method == 'srf':
                result = parse_selfrefine_response(raw_response, question)
                visualization = create_srf_diagram(result, viz_config)
            elif reasoning_method == 'bs':
                result = parse_bs_response(raw_response, question)
                visualization = create_bs_diagram(result, viz_config)

            # Display Results
            st.success("Response generated successfully!")
            st.subheader("Raw Output")
            st.text(raw_response)

            st.subheader("Visualization")
            if visualization:
                st.code(visualization, language="mermaid")
            else:
                st.warning("Visualization could not be generated.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error processing request: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**ReasonGraph** - A tool for visualizing LLM reasoning paths.")
