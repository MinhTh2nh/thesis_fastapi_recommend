# Project Setup and Running Instructions

This document provides a step-by-step guide to set up the virtual environment, install dependencies, and run the application.

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.7 or later
- pip (Python package manager)
- uvicorn

## Setup Instructions

1. **Create a Virtual Environment**

   Run the following command to create a virtual environment named `venv`:
   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**

   Activate the virtual environment using the appropriate command for your operating system:
   - On **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - On **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies**

   Use the following command to install all required dependencies listed in `setup.txt`:
   ```bash
   pip install -r setup.txt
   ```

4. **Run the Application**

   Start the application using `uvicorn` with the following command:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   This will start the application on port 8000 and make it accessible at `http://0.0.0.0:8000/`.

## Notes

- The `--reload` flag is used for development purposes. It automatically reloads the server whenever changes are detected in the code.
- Ensure the `setup.txt` file contains all necessary dependencies for the project.

## Troubleshooting

- If you encounter issues activating the virtual environment, verify that Python and pip are correctly installed and accessible in your system's PATH.
- If dependencies fail to install, check the `setup.txt` file for typos or invalid package names.
