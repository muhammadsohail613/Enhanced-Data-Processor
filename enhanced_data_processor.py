import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
from datetime import datetime
import pickle

# Set page configuration
st.set_page_config(page_title="Enhanced Data Processor", layout="wide")

# Initialize session state variables if they don't exist
if 'saved_settings' not in st.session_state:
    st.session_state.saved_settings = {}
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# CSS to improve UI appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #4e89ae;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("Enhanced Data Processor")
st.write("Upload, analyze, modify, and transform your data with multiple processing options.")

# Sidebar for app navigation
with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio("Choose a mode:", [
        "Single File Processing", 
        "Batch Processing",
        "Saved Settings"
    ])

    # Display app information
    st.markdown("---")
    st.subheader("About")
    st.info(
        """
        This app allows you to:
        - Delete, rename, and convert columns
        - Filter and sort data
        - Handle missing values
        - Visualize data
        - Process multiple files
        - Save your settings
        """
    )

# Function to load data from file
def load_data(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_extension == "csv":
            # Add options for CSV import
            delimiter_options = [",", ";", "\t", "|"]
            delimiter = st.selectbox("Select CSV delimiter", delimiter_options, index=0)
            df = pd.read_csv(uploaded_file, delimiter=delimiter)
        elif file_extension in ["xlsx", "xls"]:
            # For Excel files, show sheet selection if multiple sheets exist
            excel_file = pd.ExcelFile(uploaded_file)
            sheets = excel_file.sheet_names
            if len(sheets) > 1:
                selected_sheet = st.selectbox("Select sheet", sheets)
            else:
                selected_sheet = sheets[0]
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Function to generate download link
def get_download_link(df, filename, file_format):
    try:
        if file_format == "csv":
            # CSV format
            delimiter_options = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}
            selected_delimiter = st.selectbox("Select CSV delimiter for export", list(delimiter_options.keys()), index=0)
            csv = df.to_csv(index=False, sep=selected_delimiter)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download Modified CSV File</a>'
            return href
        elif file_format == "xlsx":
            # Excel format
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Modified Excel File</a>'
            return href
        elif file_format == "json":
            # JSON format
            json_str = df.to_json(orient='records')
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="{filename}.json">Download Modified JSON File</a>'
            return href
        else:
            return "Unsupported file format for download"
    except Exception as e:
        return f"Error generating download link: {e}"

# Function to visualize data
def visualize_data(df):
    st.subheader("Data Visualization")
    
    # Select visualization type
    viz_type = st.selectbox(
        "Select visualization type",
        ["Histogram", "Bar Chart", "Box Plot", "Correlation Heatmap", "Scatter Plot", "Pair Plot"]
    )
    
    # Configure and create the selected visualization
    if viz_type == "Histogram":
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column for histogram", numeric_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[col].dropna(), bins=30, edgecolor='black')
            ax.set_title(f'Histogram of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for histogram.")
            
    elif viz_type == "Bar Chart":
        # For bar charts, allow categorical columns
        possible_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if possible_cols:
            col = st.selectbox("Select column for bar chart", possible_cols)
            value_counts = df[col].value_counts().head(20)  # Limit to top 20 for visibility
            
            fig, ax = plt.subplots(figsize=(12, 6))
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f'Bar Chart of {col} (Top 20 values)')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No categorical columns available for bar chart.")
            
    elif viz_type == "Box Plot":
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            selected_cols = st.multiselect("Select columns for box plot", numeric_cols, default=[numeric_cols[0]] if numeric_cols else [])
            if selected_cols:
                fig, ax = plt.subplots(figsize=(12, 6))
                df[selected_cols].boxplot(ax=ax)
                ax.set_title('Box Plot')
                ax.set_ylabel('Value')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("No numeric columns available for box plot.")
            
    elif viz_type == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f", linewidths=.5)
            ax.set_title('Correlation Heatmap')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for correlation analysis.")
            
    elif viz_type == "Scatter Plot":
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            col_x = st.selectbox("Select X-axis column", numeric_cols, index=0)
            col_y = st.selectbox("Select Y-axis column", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[col_x], df[col_y], alpha=0.5)
            ax.set_title(f'Scatter Plot: {col_x} vs {col_y}')
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            st.pyplot(fig)
        else:
            st.warning("Need at least two numeric columns for scatter plot.")
            
    elif viz_type == "Pair Plot":
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            # Allow selecting a subset of columns
            selected_cols = st.multiselect(
                "Select columns for pair plot (limit to 2-5 for better visibility)", 
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            if len(selected_cols) >= 2 and len(selected_cols) <= 5:
                # Create a sample if the dataset is large
                sample_size = min(1000, len(df))
                df_sample = df[selected_cols].sample(sample_size) if len(df) > 1000 else df[selected_cols]
                
                fig = sns.pairplot(df_sample)
                fig.fig.suptitle("Pair Plot", y=1.02)
                st.pyplot(fig.fig)
            else:
                st.warning("Please select between 2 and 5 columns for the pair plot.")
        else:
            st.warning("Need at least two numeric columns for pair plot.")

# Function to save settings
def save_current_settings(settings_name, settings_dict):
    if settings_name:
        # Save in session state
        st.session_state.saved_settings[settings_name] = settings_dict
        st.success(f"Settings '{settings_name}' saved successfully!")
        
        # Optionally save to disk for persistence
        try:
            os.makedirs('saved_settings', exist_ok=True)
            with open(f'saved_settings/{settings_name}.pkl', 'wb') as f:
                pickle.dump(settings_dict, f)
        except Exception as e:
            st.warning(f"Could not save settings to disk: {e}")

# Function to load settings from disk
def load_settings_from_disk():
    try:
        os.makedirs('saved_settings', exist_ok=True)
        settings_files = glob.glob('saved_settings/*.pkl')
        for file in settings_files:
            settings_name = os.path.basename(file).replace('.pkl', '')
            if settings_name not in st.session_state.saved_settings:
                with open(file, 'rb') as f:
                    st.session_state.saved_settings[settings_name] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading saved settings: {e}")

# Load saved settings at startup
load_settings_from_disk()

# Single File Processing Mode
if app_mode == "Single File Processing":
    st.header("Single File Processing")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Display file details
        file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size / 1024:.2f} KB"}
        st.write(file_details)
        
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Sample option
            sample_enabled = st.checkbox("Work with data sample (recommended for large datasets)", 
                                         value=len(df) > 10000)
            sample_size = st.slider("Sample size", min_value=100, max_value=min(10000, len(df)), 
                                    value=min(1000, len(df))) if sample_enabled else len(df)
            
            if sample_enabled:
                df_display = df.sample(sample_size, random_state=42)
                st.write(f"Working with a random sample of {sample_size} rows from {len(df)} total rows.")
            else:
                df_display = df
            
            # Create tabs for different operations
            tabs = st.tabs(["Data Overview", "Modify Columns", "Filter & Sort", "Handle Missing Values", 
                            "Data Visualization", "Export"])
            
            # Tab 1: Data Overview
            with tabs[0]:
                st.subheader("Data Overview")
                
                # Show data shape
                st.write(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Show column information
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isna().sum().values,
                    'Null %': (df.isna().sum().values / len(df) * 100).round(2),
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info)
                
                # Display data sample
                st.subheader("Data Sample")
                st.dataframe(df_display.head(10))
                
                # Display data description (statistical summary)
                st.subheader("Data Summary (describe)")
                st.write(df_display.describe())
            
            # Tab 2: Modify Columns
            with tabs[1]:
                st.subheader("Modify Columns")
                
                # Create three columns for the different column operations
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Delete Columns")
                    columns_to_delete = st.multiselect(
                        "Select columns to delete",
                        options=df.columns.tolist(),
                        default=None
                    )
                
                with col2:
                    st.markdown("### Rename Columns")
                    st.write("Select columns to rename")
                    
                    columns_to_rename = {}
                    rename_expander = st.expander("Column Renaming Interface")
                    with rename_expander:
                        for i, col in enumerate(df.columns):
                            if st.checkbox(f"Rename {col}", key=f"rename_cb_{i}"):
                                new_name = st.text_input(f"New name for {col}", value=col, key=f"rename_input_{i}")
                                if new_name != col:
                                    columns_to_rename[col] = new_name
                
                with col3:
                    st.markdown("### Convert Data Types")
                    st.write("Select columns to convert")
                    
                    columns_to_convert = {}
                    convert_expander = st.expander("Data Type Conversion Interface")
                    with convert_expander:
                        for i, col in enumerate(df.columns):
                            current_type = df[col].dtype
                            if st.checkbox(f"Convert {col} (current: {current_type})", key=f"convert_cb_{i}"):
                                target_type = st.selectbox(
                                    f"Convert {col} to:",
                                    ["int", "float", "str", "datetime", "category", "bool"],
                                    key=f"convert_select_{i}"
                                )
                                columns_to_convert[col] = target_type
                
                # Apply modifications if requested
                if st.button("Apply Column Modifications"):
                    modified = False
                    df_modified = df.copy()
                    
                    # Process deletions
                    if columns_to_delete:
                        df_modified = df_modified.drop(columns=columns_to_delete)
                        modified = True
                    
                    # Process renamings
                    if columns_to_rename:
                        df_modified = df_modified.rename(columns=columns_to_rename)
                        modified = True
                    
                    # Process conversions
                    if columns_to_convert:
                        for col, target_type in columns_to_convert.items():
                            try:
                                if col in df_modified.columns:  # Check if column still exists
                                    if target_type == "int":
                                        df_modified[col] = pd.to_numeric(df_modified[col], errors='coerce').fillna(0).astype(int)
                                    elif target_type == "float":
                                        df_modified[col] = pd.to_numeric(df_modified[col], errors='coerce')
                                    elif target_type == "str":
                                        df_modified[col] = df_modified[col].astype(str)
                                    elif target_type == "datetime":
                                        df_modified[col] = pd.to_datetime(df_modified[col], errors='coerce')
                                    elif target_type == "category":
                                        df_modified[col] = df_modified[col].astype('category')
                                    elif target_type == "bool":
                                        df_modified[col] = df_modified[col].astype(bool)
                                modified = True
                            except Exception as e:
                                st.error(f"Error converting {col} to {target_type}: {e}")
                    
                    if modified:
                        df = df_modified  # Update the main dataframe
                        st.success("Column modifications applied successfully!")
                        
                        # Show modified dataframe
                        st.subheader("Modified Data Preview")
                        st.dataframe(df.head(10))
                        
                        # Show summary statistics
                        st.write(f"Original shape: {df_display.shape[0]} rows × {df_display.shape[1]} columns")
                        st.write(f"Modified shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    else:
                        st.warning("No modifications were specified.")
            
            # Tab 3: Filter & Sort
            with tabs[2]:
                st.subheader("Filter & Sort Data")
                
                # Filtering section
                st.markdown("### Filter Data")
                
                filter_expander = st.expander("Filtering Interface")
                with filter_expander:
                    # Initialize list to store filter conditions
                    filter_conditions = []
                    
                    # Add filters dynamically
                    num_filters = st.number_input("Number of filter conditions", min_value=0, max_value=5, value=1)
                    
                    for i in range(num_filters):
                        st.markdown(f"**Filter {i+1}**")
                        
                        # Three columns for filter construction
                        fc1, fc2, fc3 = st.columns(3)
                        
                        with fc1:
                            filter_col = st.selectbox(f"Column {i+1}", df.columns, key=f"filter_col_{i}")
                        
                        with fc2:
                            # Adjust operators based on column data type
                            if df[filter_col].dtype == 'object' or df[filter_col].dtype.name == 'category':
                                operators = ["equals", "not equals", "contains", "starts with", "ends with", "is null", "is not null"]
                            elif pd.api.types.is_numeric_dtype(df[filter_col]):
                                operators = ["equals", "not equals", "greater than", "less than", "greater or equal", "less or equal", "is null", "is not null"]
                            else:
                                operators = ["equals", "not equals", "is null", "is not null"]
                                
                            filter_op = st.selectbox(f"Operator {i+1}", operators, key=f"filter_op_{i}")
                        
                        with fc3:
                            # Only show value input if the operator needs a value
                            if filter_op not in ["is null", "is not null"]:
                                if df[filter_col].dtype == 'object' or df[filter_col].dtype.name == 'category':
                                    # For categorical/string, show unique values dropdown
                                    unique_vals = df[filter_col].dropna().unique()
                                    if len(unique_vals) <= 50:  # Only for reasonably sized value sets
                                        filter_val = st.selectbox(f"Value {i+1}", [""] + list(unique_vals), key=f"filter_val_select_{i}")
                                    else:
                                        filter_val = st.text_input(f"Value {i+1}", key=f"filter_val_text_{i}")
                                elif pd.api.types.is_numeric_dtype(df[filter_col]):
                                    # For numeric, show number input
                                    min_val = float(df[filter_col].min()) if not df[filter_col].empty else 0
                                    max_val = float(df[filter_col].max()) if not df[filter_col].empty else 100
                                    filter_val = st.number_input(f"Value {i+1}", min_value=min_val, max_value=max_val, 
                                                               value=min_val, key=f"filter_val_num_{i}")
                                else:
                                    # Default text input for other types
                                    filter_val = st.text_input(f"Value {i+1}", key=f"filter_val_default_{i}")
                            else:
                                filter_val = None
                        
                        # Build the filter condition
                        if filter_op == "equals":
                            filter_conditions.append(f"df['{filter_col}'] == '{filter_val}'")
                        elif filter_op == "not equals":
                            filter_conditions.append(f"df['{filter_col}'] != '{filter_val}'")
                        elif filter_op == "greater than":
                            filter_conditions.append(f"df['{filter_col}'] > {filter_val}")
                        elif filter_op == "less than":
                            filter_conditions.append(f"df['{filter_col}'] < {filter_val}")
                        elif filter_op == "greater or equal":
                            filter_conditions.append(f"df['{filter_col}'] >= {filter_val}")
                        elif filter_op == "less or equal":
                            filter_conditions.append(f"df['{filter_col}'] <= {filter_val}")
                        elif filter_op == "contains":
                            filter_conditions.append(f"df['{filter_col}'].astype(str).str.contains('{filter_val}', na=False)")
                        elif filter_op == "starts with":
                            filter_conditions.append(f"df['{filter_col}'].astype(str).str.startswith('{filter_val}', na=False)")
                        elif filter_op == "ends with":
                            filter_conditions.append(f"df['{filter_col}'].astype(str).str.endswith('{filter_val}', na=False)")
                        elif filter_op == "is null":
                            filter_conditions.append(f"df['{filter_col}'].isna()")
                        elif filter_op == "is not null":
                            filter_conditions.append(f"df['{filter_col}'].notna()")
                
                # Sorting section
                st.markdown("### Sort Data")
                
                sort_expander = st.expander("Sorting Interface")
                with sort_expander:
                    sort_columns = st.multiselect("Sort by columns", df.columns)
                    
                    # For each selected column, ask for sort direction
                    sort_ascending = []
                    for col in sort_columns:
                        is_ascending = st.radio(f"Sort {col}", ["Ascending", "Descending"], 
                                               index=0, key=f"sort_dir_{col}") == "Ascending"
                        sort_ascending.append(is_ascending)
                
                # Apply filters and sorting
                if st.button("Apply Filters and Sorting"):
                    df_filtered = df.copy()
                    
                    # Apply filters
                    if filter_conditions:
                        try:
                            combined_filter = " & ".join(filter_conditions)
                            df_filtered = df_filtered[eval(combined_filter)]
                            st.success(f"Filtering applied successfully! {len(df_filtered)} rows matched your criteria.")
                        except Exception as e:
                            st.error(f"Error applying filters: {e}")
                    
                    # Apply sorting
                    if sort_columns:
                        try:
                            df_filtered = df_filtered.sort_values(by=sort_columns, ascending=sort_ascending)
                            st.success("Sorting applied successfully!")
                        except Exception as e:
                            st.error(f"Error applying sorting: {e}")
                    
                    # Show filtered and sorted data
                    st.subheader("Filtered and Sorted Data")
                    st.dataframe(df_filtered.head(50))
                    
                    # Update main dataframe
                    df = df_filtered
            
            # Tab 4: Handle Missing Values
            with tabs[3]:
                st.subheader("Handle Missing Values")
                
                # Calculate missing values statistics
                missing_stats = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Count': df.isna().sum().values,
                    'Missing %': (df.isna().sum().values / len(df) * 100).round(2)
                }).sort_values('Missing %', ascending=False)
                
                # Display missing values statistics
                st.markdown("### Missing Values Statistics")
                st.dataframe(missing_stats)
                
                # Visualize missing values
                st.markdown("### Missing Values Visualization")
                try:
                    if not df.empty:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        ax.barh(missing_stats['Column'], missing_stats['Missing %'])
                        ax.set_title('Missing Values Percentage by Column')
                        ax.set_xlabel('Missing Values (%)')
                        ax.set_ylabel('Column')
                        plt.tight_layout()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error visualizing missing values: {e}")
                
                # Missing value handling options
                st.markdown("### Handle Missing Values")
                
                # Select columns for missing value handling
                cols_with_missing = missing_stats[missing_stats['Missing Count'] > 0]['Column'].tolist()
                
                if cols_with_missing:
                    selected_cols = st.multiselect(
                        "Select columns with missing values to handle",
                        options=cols_with_missing
                    )
                    
                    # For each selected column, provide handling options
                    if selected_cols:
                        handling_methods = {}
                        
                        for col in selected_cols:
                            st.markdown(f"**{col}**")
                            
                            # Determine data type for appropriate options
                            col_type = df[col].dtype
                            
                            if pd.api.types.is_numeric_dtype(col_type):
                                method = st.selectbox(
                                    f"How to handle missing values in {col}",
                                    ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode",
                                     "Fill with zero", "Fill with custom value"],
                                    key=f"method_{col}"
                                )
                                
                                if method == "Fill with custom value":
                                    custom_val = st.number_input(f"Custom value for {col}", key=f"custom_{col}")
                                    handling_methods[col] = ('custom', custom_val)
                                else:
                                    handling_methods[col] = (method, None)
                            
                            else:  # Non-numeric columns
                                method = st.selectbox(
                                    f"How to handle missing values in {col}",
                                    ["Drop rows", "Fill with mode", "Fill with empty string", 
                                     "Fill with custom value"],
                                    key=f"method_{col}"
                                )
                                
                                if method == "Fill with custom value":
                                    custom_val = st.text_input(f"Custom value for {col}", key=f"custom_{col}")
                                    handling_methods[col] = ('custom', custom_val)
                                else:
                                    handling_methods[col] = (method, None)
                        
                        # Apply handling methods
                        if st.button("Apply Missing Value Handling"):
                            df_handled = df.copy()
                            
                            for col, (method, custom_val) in handling_methods.items():
                                try:
                                    if method == "Drop rows":
                                        df_handled = df_handled.dropna(subset=[col])
                                    elif method == "Fill with mean":
                                        df_handled[col] = df_handled[col].fillna(df_handled[col].mean())
                                    elif method == "Fill with median":
                                        df_handled[col] = df_handled[col].fillna(df_handled[col].median())
                                    elif method == "Fill with mode":
                                        df_handled[col] = df_handled[col].fillna(df_handled[col].mode()[0] if not df_handled[col].mode().empty else '')
                                    elif method == "Fill with zero":
                                        df_handled[col] = df_handled[col].fillna(0)
                                    elif method == "Fill with empty string":
                                        df_handled[col] = df_handled[col].fillna('')
                                    elif method == "Fill with custom value":
                                        df_handled[col] = df_handled[col].fillna(custom_val)
                                except Exception as e:
                                    st.error(f"Error handling missing values in {col}: {e}")
                            
                            # Update main dataframe
                            df = df_handled
                            
                            # Show handled data
                            st.success("Missing values handled successfully!")
                            st.subheader("Data After Handling Missing Values")
                            st.dataframe(df.head(10))
                            
                            # Update missing values statistics
                            new_missing_stats = pd.DataFrame({
                                'Column': df.columns,
                                'Missing Count': df.isna().sum().values,
                                'Missing %': (df.isna().sum().values / len(df) * 100).round(2)
                            }).sort_values('Missing %', ascending=False)
                            
                            st.subheader("Updated Missing Values Statistics")
                            st.dataframe(new_missing_stats)
                else:
                    st.info("No columns with missing values found in the dataset.")
            
            # Tab 5: Data Visualization
            with tabs[4]:
                visualize_data(df)
            
            # Tab 6: Export
            with tabs[5]:
                st.subheader("Export Processed Data")
                
                # Generate filename for download
                original_filename = uploaded_file.name.split(".")[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                modified_filename = f"{original_filename}_processed_{timestamp}"
                
                # Let user choose output format
                output_format = st.radio(
                    "Select output file format",
                    options=["csv", "xlsx", "json"],
                    horizontal=True
                )
                
                # Save settings option
                save_settings = st.checkbox("Save processing settings for future use")
                
                if save_settings:
                    settings_name = st.text_input("Settings name", value=f"Settings_{timestamp}")
                    
                    # Collect all current settings
                    current_settings = {
                        "columns_to_delete": columns_to_delete if 'columns_to_delete' in locals() else [],
                        "columns_to_rename": columns_to_rename if 'columns_to_rename' in locals() else {},
                        "columns_to_convert": columns_to_convert if 'columns_to_convert' in locals() else {},
                        # Add other settings here
                    }
                    
                    if st.button("Save Current Settings"):
                        save_current_settings(settings_name, current_settings)
                
                # Create download button
                st.markdown(
                    get_download_link(df, modified_filename, output_format),
                    unsafe_allow_html=True
                )
                
                # Display summary statistics
                st.subheader("Final Data Summary")
                st.write(f"Original rows: {len(df_display)}")
                st.write(f"Final rows: {len(df)}")
                st.write(f"Original columns: {len(df_display.columns)}")
                st.write(f"Final columns: {len(df.columns)}")
                
                # Display final data sample
                st.subheader("Final Data Sample")
                st.dataframe(df.head(10))

# Batch Processing Mode
elif app_mode == "Batch Processing":
    st.header("Batch Processing")
    
    st.info("Batch processing allows you to apply the same operations to multiple files at once.")
    
    # Select a saved setting to apply
    if st.session_state.saved_settings:
        st.subheader("Select Processing Settings")
        selected_settings = st.selectbox(
            "Choose saved settings to apply",
            options=list(st.session_state.saved_settings.keys())
        )
        
        # Show selected settings details
        if selected_settings:
            st.write("Selected settings:")
            st.json(st.session_state.saved_settings[selected_settings])
    else:
        st.warning("No saved settings found. Please create settings in Single File Processing mode first.")
        selected_settings = None
    
    # Upload multiple files
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader("Choose CSV or Excel files", 
                                     type=["csv", "xlsx", "xls"], 
                                     accept_multiple_files=True)
    
    if uploaded_files and selected_settings:
        st.write(f"Uploaded {len(uploaded_files)} files:")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size / 1024:.2f} KB)")
        
        # Process all files button
        if st.button("Process All Files"):
            settings = st.session_state.saved_settings[selected_settings]
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Process each file
            processed_files = []
            for i, file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    # Load file
                    df = load_data(file)
                    
                    if df is not None:
                        # Apply settings
                        # 1. Delete columns
                        if "columns_to_delete" in settings and settings["columns_to_delete"]:
                            # Only delete columns that exist in this file
                            cols_to_delete = [col for col in settings["columns_to_delete"] if col in df.columns]
                            if cols_to_delete:
                                df = df.drop(columns=cols_to_delete)
                        
                        # 2. Rename columns
                        if "columns_to_rename" in settings and settings["columns_to_rename"]:
                            # Only rename columns that exist in this file
                            rename_dict = {k: v for k, v in settings["columns_to_rename"].items() if k in df.columns}
                            if rename_dict:
                                df = df.rename(columns=rename_dict)
                        
                        # 3. Convert column types
                        if "columns_to_convert" in settings and settings["columns_to_convert"]:
                            for col, target_type in settings["columns_to_convert"].items():
                                if col in df.columns:
                                    try:
                                        if target_type == "int":
                                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                                        elif target_type == "float":
                                            df[col] = pd.to_numeric(df[col], errors='coerce')
                                        elif target_type == "str":
                                            df[col] = df[col].astype(str)
                                        elif target_type == "datetime":
                                            df[col] = pd.to_datetime(df[col], errors='coerce')
                                        elif target_type == "category":
                                            df[col] = df[col].astype('category')
                                        elif target_type == "bool":
                                            df[col] = df[col].astype(bool)
                                    except Exception as e:
                                        st.warning(f"Error converting {col} to {target_type} in {file.name}: {e}")
                        
                        # Generate output filename
                        original_filename = file.name.split(".")[0]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"{original_filename}_processed_{timestamp}"
                        
                        # Generate download link
                        output_format = "csv"  # Default format, could be made configurable
                        download_link = get_download_link(df, output_filename, output_format)
                        
                        # Add to processed files list
                        processed_files.append({
                            "filename": file.name,
                            "processed_rows": len(df),
                            "processed_columns": len(df.columns),
                            "download_link": download_link
                        })
                        
                        # Add to session state
                        st.session_state.processed_files = processed_files
                
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
            
            # Show completion message
            st.success(f"Successfully processed {len(processed_files)} out of {len(uploaded_files)} files!")
            
            # Display download links for processed files
            st.subheader("Download Processed Files")
            for file_info in processed_files:
                st.write(f"**{file_info['filename']}** - {file_info['processed_rows']} rows × {file_info['processed_columns']} columns")
                st.markdown(file_info['download_link'], unsafe_allow_html=True)

# Saved Settings Mode
elif app_mode == "Saved Settings":
    st.header("Saved Settings")
    
    if st.session_state.saved_settings:
        st.write(f"You have {len(st.session_state.saved_settings)} saved settings:")
        
        # Display all saved settings
        for setting_name, settings in st.session_state.saved_settings.items():
            with st.expander(f"Setting: {setting_name}"):
                st.json(settings)
                
                # Add option to delete this setting
                if st.button(f"Delete '{setting_name}'", key=f"delete_{setting_name}"):
                    del st.session_state.saved_settings[setting_name]
                    try:
                        # Remove from disk if exists
                        if os.path.exists(f'saved_settings/{setting_name}.pkl'):
                            os.remove(f'saved_settings/{setting_name}.pkl')
                    except Exception as e:
                        st.error(f"Error deleting saved setting file: {e}")
                    st.experimental_rerun()
        
        # Option to export all settings as JSON
        if st.button("Export All Settings as JSON"):
            settings_json = json.dumps(st.session_state.saved_settings, indent=4)
            b64 = base64.b64encode(settings_json.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="data_processor_settings.json">Download Settings JSON</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Option to import settings from JSON
        st.subheader("Import Settings")
        uploaded_settings = st.file_uploader("Upload settings JSON file", type=["json"])
        if uploaded_settings:
            try:
                imported_settings = json.loads(uploaded_settings.getvalue().decode())
                if isinstance(imported_settings, dict):
                    for name, setting in imported_settings.items():
                        st.session_state.saved_settings[name] = setting
                    st.success(f"Successfully imported {len(imported_settings)} settings!")
                    
                    # Also save to disk
                    os.makedirs('saved_settings', exist_ok=True)
                    for name, setting in imported_settings.items():
                        with open(f'saved_settings/{name}.pkl', 'wb') as f:
                            pickle.dump(setting, f)
                else:
                    st.error("Invalid settings file format.")
            except Exception as e:
                st.error(f"Error importing settings: {e}")
    else:
        st.info("No saved settings found. You can create settings in the Single File Processing mode.")

# Footer
st.markdown("---")
st.markdown("### About Enhanced Data Processor")
st.markdown("""
This application helps you process, analyze, and transform tabular data with multiple features:
- Delete, rename, and convert columns
- Filter and sort data based on conditions
- Handle missing values with various strategies
- Visualize data with plots and charts
- Process multiple files with the same settings
- Save and reuse processing settings
""")