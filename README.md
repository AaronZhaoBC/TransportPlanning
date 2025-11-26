# Steel Loading Planner

## Project Overview

Steel Loading Planner is an intelligent steel loading optimization tool based on historical data and OpenAI models. The system learns from historical loading patterns, analyzes physical attributes of steel items (dimensions, weight, shape, etc.), and generates efficient loading plans.

## Key Features

- 📊 **Historical Data Analysis**: Extracts frequent loading combination patterns from historical loading records
- 🤖 **AI-Assisted Decision Making**: Uses OpenAI models to generate loading plans based on physical attributes and historical patterns
- 📐 **Physical Attribute Analysis**: Automatically reads and analyzes steel attributes such as dimensions, weight, quantity, and shape
- 📁 **File Upload Support**: Supports uploading steel data files in Excel (.xlsx, .xls) or CSV formats
- 📥 **Result Export**: Exports loading plans as CSV files with the same format as input files
- 🎯 **Intelligent Parsing**: Automatically extracts loading plans from natural language responses

## Installation Requirements

### Dependencies

Ensure the following Python packages are installed:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- `pandas >= 2.0.0` - Data processing
- `openai >= 1.0.0` - OpenAI API calls
- `streamlit >= 1.28.0` - Web interface
- `openpyxl >= 3.1.0` - Excel file reading

### System Requirements

- Python 3.7+
- Valid OpenAI API key (default key is built-in, but you can configure your own)

## Usage Guide

### 1. Launch the Application

Run the following command in the terminal to start the Streamlit application:

```bash
streamlit run chatbot_trial.py
```

The application will automatically open in your browser (usually at `http://localhost:8501`)

### 2. Configuration Settings

Configure settings in the "Configuration" section of the sidebar:

- **OpenAI API key**: Enter your OpenAI API key (default key is built-in)
- **Generation temperature**: Controls the randomness of model output (0.0-1.0, recommended: 0.1)
- **Provide historical baseline to OpenAI**: Whether to provide historical baseline plan to the model
- **Number of frequent combinations to summarise**: Number of frequent combinations to summarize (3-200)
- **Show historical summary**: Whether to display historical data summary

Advanced settings (hidden by default):
- **Historical data file**: Path to historical data file (default: data.xlsx)
- **OpenAI model**: OpenAI model to use (default: gpt-4o-mini)

### 3. Upload Data File

1. Click "Upload steel items file" to upload a file containing steel item information
2. Supported formats: `.xlsx`, `.xls`, `.csv`
3. The file must contain the following columns:
   - **Required column**: `ITEM_NO` (steel item number)
   - **Optional columns** (physical attributes):
     - `LENGTH` - Length
     - `WIDTH` - Width
     - `HEIGHT` - Height
     - `DIAMETER` - Diameter
     - `FG_PRODUCTION_WT_KG` - Weight (kg)
     - `ORDER_PIECES` - Quantity/Pieces
     - `WEIGHT` - Weight (ton)
     - `SHAPE` - Shape

### 4. View Extracted Data

After uploading the file, the system will:
- Display the number of successfully loaded steel items
- Show data preview (first 10 rows)
- Display extracted physical attributes table
- Automatically aggregate pieces and weight for duplicate ITEM_NOs

### 5. Generate Loading Plan

1. Click the "Generate plan with OpenAI" button
2. The system will:
   - Analyze historical loading patterns
   - Consider physical attributes of steel items
   - Use OpenAI model to generate loading plan
   - Display loading suggestions in natural language format

### 6. View and Download Results

After generating the plan, you can:

- **View plan**: Check the model-generated loading plan in the "Plan suggested by OpenAI" section
- **View physical attributes**: Expand "Physical attributes used for planning" to view attribute data used
- **Preview output table**: Expand "Preview output table" to view the formatted loading plan table
- **Download CSV**: Click "Download as CSV" to download a CSV file with the same format as the input file
- **Download JSON**: Click "Download as JSON" to download a JSON file containing complete information

## Input File Format

### Required Columns

- `ITEM_NO`: Steel item number (string, required)

### Optional Columns (Physical Attributes)

| Column Name | Type | Description |
|-------------|------|-------------|
| `LENGTH` | Numeric | Steel length (mm) |
| `WIDTH` | Numeric | Steel width (mm) |
| `HEIGHT` | Numeric | Steel height (mm) |
| `DIAMETER` | Numeric | Steel diameter (mm) |
| `FG_PRODUCTION_WT_KG` | Numeric | Weight (kg) |
| `ORDER_PIECES` | Integer | Quantity/Pieces |
| `WEIGHT` | Numeric | Weight (ton) |
| `SHAPE` | String | Steel shape |

### Example File Structure

```csv
ITEM_NO,LENGTH,WIDTH,HEIGHT,FG_PRODUCTION_WT_KG,ORDER_PIECES,WEIGHT,SHAPE
ITEM001,5000,2000,1500,1500.5,10,1.5,shape1
ITEM002,3000,1000,800,800.2,5,0.8,shape2
```

## Output File Format

### CSV Output

The output CSV file contains the following columns:
- `LOAD_NO`: Loading number (format: LOAD_0001, LOAD_0002...)
- `ITEM_NO`: Steel item number
- All other columns from the input file (physical attributes, etc.)

Rows with the same `LOAD_NO` indicate that these steel items are loaded on the same truck.

### JSON Output

The JSON file contains:
- `response_text`: Raw response from OpenAI model
- `baseline_plan`: Baseline plan based on historical data
- `item_attributes`: Physical attributes of all steel items
- `output_table`: Formatted loading plan table

## How It Works

### 1. Historical Data Analysis

The system learns from `data.xlsx`:
- Frequent steel combination patterns
- Co-occurrence frequency of steel pairs
- Statistical information of physical attributes
- Shape distribution

### 2. Physical Attribute Extraction

For uploaded files:
- Automatically extracts physical attributes for each steel item
- For duplicate ITEM_NOs, automatically aggregates pieces and weight
- Dimensions and shape use the first non-null value

### 3. Loading Plan Generation

The system generates plans using the following information:
- Historical loading patterns
- Steel physical attributes (dimensions, weight, shape)
- Weight and space constraints
- Intelligent reasoning from OpenAI model

### 4. Result Parsing and Formatting

- Extracts loading plan from natural language response
- Assigns LOAD_NO to each load
- Preserves all physical attribute information
- Generates output with the same format as input file

## Frequently Asked Questions

### Q: How to modify the historical data file path?

A: Expand "Advanced model & data settings" in the sidebar and modify the "Historical data file" field.

### Q: What if the uploaded file is missing some physical attributes?

A: The system will try to find missing attributes from historical data. If not found in historical data, the attribute will be displayed as "N/A".

### Q: How are duplicate ITEM_NOs handled?

A: The system automatically aggregates pieces and weight for the same ITEM_NO, and uses the first non-null value for dimensions and shape.

### Q: What if the generated plan doesn't meet expectations?

A: You can try:
- Adjusting the "Generation temperature" parameter
- Increasing "Number of frequent combinations to summarise"
- Checking if physical attributes in the uploaded file are complete

### Q: Which OpenAI models are supported?

A: Default is `gpt-4o-mini`, but you can modify it in advanced settings to other models (such as `gpt-4`, `gpt-3.5-turbo`, etc.).

## Important Notes

1. **Data File Format**: Ensure uploaded file column names match the historical data file
2. **API Key**: Although a default key is built-in, it's recommended to use your own OpenAI API key for security
3. **File Size**: It's recommended not to upload files that are too large to avoid affecting processing speed
4. **Network Connection**: Generating plans requires calling OpenAI API, ensure network connection is normal

## Technical Support

If you encounter issues or have suggestions, please check:
1. Whether dependencies are correctly installed
2. Whether data file format is correct
3. Whether OpenAI API key is valid
4. Whether network connection is normal
