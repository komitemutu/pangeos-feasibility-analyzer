import streamlit as st
import os
import sys

# Add the parent directory to the path so we can import our app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and run the main app
from app import *

# This is required for Vercel deployment
if __name__ == "__main__":
    pass
