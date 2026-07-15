import pandas as pd
import json
import os

def analyze_dataset(file_path):
    """
    Reads a CSV file and extracts initial metadata for the EDA dashboard.
    This function acts as the core data processing engine before we add APIs.
    """
    if not os.path.exists(file_path):
        return {"status": "error", "message": "File not found"}
    
    try:
        # Load the dataset using Pandas
        df = pd.read_csv(file_path)
        
        # 1. Basic Shape (Rows and Columns)
        rows, cols = df.shape
        
        # 2. Identify Column Types (Numeric vs Categorical)
        column_types = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                column_types[col] = "numeric"
            else:
                column_types[col] = "categorical"
        
        # 3. Count Missing Values per column
        missing_values = df.isnull().sum().to_dict()
        
        # 4. Data Preview (First 5 rows)
        # Note: We replace Pandas NaN with None so it can be safely converted to JSON later by FastAPI
        preview_df = df.head().where(pd.notnull(df.head()), None)
        preview = preview_df.to_dict(orient='records')
        
        # 5. Summary Statistics for numerical columns (mean, min, max, etc.)
        summary_df = df.describe().where(pd.notnull(df.describe()), None)
        summary_stats = summary_df.to_dict()

        # Return everything as a structured dictionary ready for a REST API
        return {
            "status": "success",
            "filename": os.path.basename(file_path),
            "total_rows": rows,
            "total_columns": cols,
            "column_types": column_types,
            "missing_values": missing_values,
            "preview": preview,
            "summary_stats": summary_stats
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# Testing Block (To ensure our engine works)
# ==========================================
if __name__ == "__main__":
    # Create a quick dummy CSV to test our function locally
    dummy_data = {
        "Age": [25, 30, None, 45, 50],
        "Salary": [50000, 60000, 75000, None, 90000],
        "Department": ["IT", "HR", "IT", "Marketing", None]
    }
    df_dummy = pd.DataFrame(dummy_data)
    test_filename = "dummy_test.csv"
    df_dummy.to_csv(test_filename, index=False)
    
    print(f"--- Analyzing {test_filename} ---")
    
    # Run our function
    result = analyze_dataset(test_filename)
    
    # Print the output in a nice readable JSON format
    print(json.dumps(result, indent=2))
    
    # Clean up the test file
    if os.path.exists(test_filename):
        os.remove(test_filename)