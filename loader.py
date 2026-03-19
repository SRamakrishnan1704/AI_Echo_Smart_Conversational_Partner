import pandas as pd
import os

file_path = "D:\\Ramakrishnan S\\Guvi\\Visual studio\\My Project foler\\Smartest_Conversational_Partner\\Data\\chatgpt_style_reviews_dataset.xlsx"

def load_data(file_path):
    """
    Load the dataset from the specified file path.

    Parameters:
    file_path (str): The path to the dataset file.

    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    
    try:
        data = pd.read_excel(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
    
if __name__ == "__main__":
    data = load_data(file_path)
    if data is not None:
        print(data.head())
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
