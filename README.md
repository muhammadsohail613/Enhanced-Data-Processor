# Enhanced Data Processor

A powerful Streamlit application for data processing, analysis, and transformation. This tool allows users to upload CSV or Excel files, modify columns, filter data, handle missing values, visualize data, and export the processed results.

![Enhanced Data Processor Screenshot](screenshots/app_screenshot.png)

## Features

- **Column Management**: Delete, rename, and convert data types of columns
- **Data Filtering**: Apply multiple filter conditions with various operators based on column data types
- **Custom Sorting**: Sort data by multiple columns with ascending or descending options
- **Missing Value Handling**: Detect, visualize, and handle missing values with various strategies
- **Data Visualization**: Create histograms, bar charts, box plots, correlation heatmaps, scatter plots, and pair plots
- **Data Sampling**: Work with samples of large datasets to improve performance
- **Batch Processing**: Apply the same transformations to multiple files at once
- **Save Settings**: Save processing configurations for future use and apply them to other files
- **Multiple Format Support**: Import from and export to CSV, Excel, and JSON formats
- **Interactive UI**: User-friendly interface with tabs for different operations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/enhanced-data-processor.git
cd enhanced-data-processor
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run enhanced_data_processor.py
```

## Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
xlsxwriter
openpyxl
```

## Usage

### Single File Processing

1. Upload a CSV or Excel file using the file uploader
2. Navigate through the tabs to perform different operations:
   - **Data Overview**: View data summary and statistics
   - **Modify Columns**: Delete, rename, or convert column data types
   - **Filter & Sort**: Apply custom filters and sorting
   - **Handle Missing Values**: Detect and handle missing data
   - **Data Visualization**: Create various charts and plots
   - **Export**: Save the processed data in your preferred format

### Batch Processing

1. Create and save settings in Single File Processing mode
2. Switch to Batch Processing mode
3. Select the saved settings to apply
4. Upload multiple files to process them with the same settings

### Saved Settings

- Save your processing configurations for future use
- Export settings as JSON files
- Import settings from JSON files

## Example Workflow

1. Upload a CSV file of customer data
2. Delete sensitive columns like personal identifiers
3. Rename columns to a more readable format
4. Convert date columns to datetime format
5. Filter out rows with missing critical fields
6. Visualize the distribution of key metrics
7. Export the cleaned data as an Excel file

## Screenshots

The application provides an intuitive user interface with different sections for specific data processing tasks:

![Enhanced Data Processor Screenshot](screenshots/app_screenshot.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Built with [Streamlit](https://streamlit.io/)
- Data processing powered by [Pandas](https://pandas.pydata.org/)
- Visualizations created with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
