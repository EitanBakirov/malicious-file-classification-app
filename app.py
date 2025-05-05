import streamlit as st
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from detect_malware import detect_malware, train_and_save_model

st.set_page_config(page_title="Malware Detection Tool", page_icon="🛡️", layout="wide")

def model_exists():
    """Check if the trained model file exists"""
    model_path = "model/trained_model.pkl"
    return os.path.exists(model_path)

def main():
    # Initialize session state variables if they don't exist
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    st.title("🛡️ Malware Detection Tool")
    
    # Add project explanation as an expander in the main page
    with st.expander("ℹ️ About This Project", expanded=False):
        st.markdown("""
        ### Project Overview
        This project implements a complete machine learning pipeline for malware detection:
        
        - **Exploratory Data Analysis (EDA)**: Comprehensive analysis of file characteristics and patterns
        - **Preprocessing**: Handling missing values, categorical features, outliers, and dimensionality reduction
        - **Model Selection**: Evaluation of multiple algorithms including KNN, Logistic Regression, ANN, and Random Forest
        - **Evaluation**: Detailed assessment using various metrics including AUC, precision, and recall
        
        The **Random Forest** model was selected as the best performer with an **AUC of 0.9753**. \n
        The dataset consists of static analysis features from **60,000 benign and malicious files**.
        Each time the model is trained, predictions are made on a separate test set of **18,000 files** and saved to **results.csv**.
        
        ---
        
        📓 [**View the complete ML pipeline notebook on GitHub**](https://github.com/eitanbakirov/malicious-file-classification/blob/main/notebook.ipynb)
        """)

    st.write("""
    ## Upload a file to analyze
    This tool uses machine learning to detect whether a file is likely malicious based on static analysis.
    """)
    
    # Sidebar with options
    with st.sidebar:
        st.header("Options")
        threshold = st.slider(
            "Detection Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            help="Probability threshold for classifying a file as malicious"
        )

        if st.button("Train Model"):
            with st.spinner("Training model... This may take a while"):
                train_and_save_model()
            st.success("Model trained and saved successfully!")
        
        # Display model information
        st.markdown("---")
        st.subheader("Model Information")
        if model_exists():
            st.info("✅ **Random Forest** model is trained and ready to use!")
            
            # Add model evaluation metrics
            with st.expander("Model Performance", expanded=False):
                model_metrics = {
                    "Mean AUC": 0.9753,
                    "Precision": 0.93,
                    "Recall": 0.91
                }

                # Display metrics in a clean format
                for metric, value in model_metrics.items():
                    st.markdown(f"**{metric}:** {value}")

            # Add model parameters in an expander
            with st.expander("Model Parameters", expanded=False):
                model_params = {
                    "Number of trees": 300,
                    "Max depth": "None (unlimited)",
                    "Min samples split": 10,
                    "Min samples leaf": 4,
                    "Feature selection": "All features",
                    "Bootstrap": "Yes",
                    "Class weight": "Balanced"
                }
                
                # Display parameters in a clean format
                for param, value in model_params.items():
                    st.markdown(f"**{param}:** {value}")
        else:
            st.warning("⚠️ No trained model found. Please click 'Train Model' to create a new model before analyzing files.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file to analyze", type=None)
    
    if uploaded_file is not None:
        # First check if model exists
        if not model_exists():
            st.warning("No trained model found. Training a Random Forest model now...")
            with st.spinner("Training model... This may take a while"):
                train_and_save_model()
                # Add this line to force a rerun after model training
                st.session_state.model_trained = True
                st.success("Model trained and ready to use!")
                # Force a rerun to update the sidebar
                st.rerun()
        
        try:
            # Save uploaded file temporarily and get its path
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name
            
            try:
                # Show a spinner while processing
                with st.spinner("Analyzing file..."):
                    # Run the malware detection only once
                    result = detect_malware(temp_path, threshold=threshold, original_filename=uploaded_file.name)
                
                # Check if there was an error
                if "error" in result:
                    st.error(f"Analysis failed: {result['error']}")
                else:
                    # Display the results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Results")
                        st.write(f"**File:** {result['file']}")
                        st.write(f"**SHA256:** {result['sha256']}")
                        
                        # Display prediction with more appealing styling
                        prob = result['malicious_probability']
                        
                        # Create a nicer prediction display
                        if result['prediction'] == "Malicious":
                            pred_color = "red"
                            emoji = "⚠️"
                        else:
                            pred_color = "green"
                            emoji = "✅"
                            
                        # Use custom HTML/markdown to make it more appealing
                        st.markdown(f"""
                        <div style="padding: 15px; border-radius: 8px; background-color: rgba({255 if pred_color == 'red' else 0}, {255 if pred_color == 'green' else 0}, 0, 0.2); margin-bottom: 15px; border-left: 5px solid {pred_color};">
                            <h2 style="color: {pred_color}; margin: 0; display: flex; align-items: center;">{emoji} {result['prediction']}</h2>
                            <p style="margin-top: 5px; font-size: 1.2em;">Malicious Probability: <strong>{prob:.2%}</strong></p>
                            <small style="opacity: 0.8;">Analysis performed using Random Forest classifier</small>
                        </div>
                        """, unsafe_allow_html=True)

                    # Show feature details in an expander
                    with st.expander("Feature Details"):
                        # Instead of using dataframe, which requires Arrow serialization,
                        # let's use a simpler approach with a dictionary display
                        features_dict = result['features'].copy()
                        
                        # Dictionary of feature explanations for hover text
                        feature_explanations = {
                            "sha256": "A unique identifier for the file",
                            "size": "Size of file on disk",
                            "vsize": "Virtual size – size of the file image when loaded into memory",
                            "imports": "Number of imported functions",
                            "exports": "Number of exported functions",
                            "has_debug": "Whether a file has a debug section",
                            "has_relocations": "Whether a file has relocations",
                            "has_resources": "Whether a file has resources",
                            "has_signature": "Whether a file has a signature",
                            "has_tls": "Whether a file has thread local storage",
                            "symbols": "Number of symbols",
                            "numstrings": "The number of printable strings that are at least five printable characters long",
                            "paths": "Number of strings that begin with C:\\ (case insensitive) that may indicate a path",
                            "urls": "Number of occurrences of http:// or https:// (case insensitive) that may indicate a URL",
                            "registry": "The number of occurrences of HKEY_ that may indicate a registry key",
                            "MZ": "The number of occurrences of the short string MZ that may provide weak evidence of a Windows PE dropper or bundled executables",
                            "printables": "Number of printable characters",
                            "avlength": "Average string length",
                            "file_type_trid": "The file type with the highest probability, given by TRID",
                            "file_type_prob_trid": "The probability of mentioned file type"
                        }
                        
                        # Display features in the specified order
                        feature_order = [
                            "sha256", "size", "vsize", "imports", "exports", "has_debug", 
                            "has_relocations", "has_resources", "has_signature", "has_tls", 
                            "symbols", "numstrings", "paths", "urls", "registry", "MZ", 
                            "printables", "avlength", "file_type_trid", "file_type_prob_trid"
                        ]
                        
                        # Create a container for the feature details
                        feature_container = st.container()

                        # Find the longest feature name to set appropriate column width
                        longest_feature = max(feature_order, key=len)
                        longest_length = len(longest_feature)
                        # Add a bit of padding for the bold formatting
                        col1_width = min(0.3, max(0.2, longest_length / 50))
                        col2_width = 1 - col1_width

                        # Display features in the specified order with hover text
                        for feature in feature_order:
                            if feature in features_dict:
                                # Get the feature value
                                value = features_dict[feature]
                                # Get the explanation (or default to empty string)
                                explanation = feature_explanations.get(feature, "")
                                
                                # Create two columns for each feature with adjusted widths
                                col1, col2 = feature_container.columns([col1_width, col2_width])
                                
                                # Format the display text
                                if explanation:
                                    col1.markdown(f"**{feature}**", help=explanation)
                                else:
                                    col1.markdown(f"**{feature}**")
                                
                                col2.write(f"{value}")

                        # Display any remaining features not in the specified order
                        remaining_features = [f for f in features_dict if f not in feature_order]

                        if remaining_features:
                            st.markdown("---")
                            st.write("**Other Features:**")
                            for feature in sorted(remaining_features):
                                col1, col2 = st.columns([col1_width, col2_width])
                                col1.markdown(f"**{feature}**")
                                col2.write(f"{features_dict[feature]}")
                        
            finally:
                # Clean up the temp file
                os.unlink(temp_path)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Print traceback for more details
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()