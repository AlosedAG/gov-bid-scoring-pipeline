import pandas as pd
import tkinter as tk
from tkinter import filedialog
from config import *

def load_populations(filepath):
    """Loads the population CSV."""
    try:
        df = pd.read_csv(filepath)
        return df.fillna('')
    except Exception as e:
        print(f"Error loading populations: {e}")
        return pd.DataFrame()

def load_landscapes(filepath):
    """Loads the landscapes CSV and extracts the first column as a list."""
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return []
        # Take the first column (Column A), drop empty rows, convert to list
        landscapes = df.iloc[:, 0].dropna().astype(str).tolist()
        # Clean whitespace and remove empty strings
        return [l.strip() for l in landscapes if l.strip()]
    except Exception as e:
        print(f"Error loading landscapes: {e}")
        return []

def select_and_load_bids():
    """Opens file dialog and loads the raw bids CSV."""
    print("Initializing file dialog...")
    
    root = tk.Tk()
    root.withdraw() 
    
    # Force window to front
    root.attributes('-topmost', True) 
    root.lift()
    root.focus_force()

    print("Please select the input CSV file in the popup window...")
    
    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select Bids CSV",
        filetypes=[("CSV files", "*.csv")]
    )
    
    root.destroy()

    if not file_path:
        print("No file selected.")
        return None, None

    print(f"File selected: {file_path}")

    try:
        # Load without header to treat by index
        df = pd.read_csv(file_path, header=None)
        return df, file_path
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None