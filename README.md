# ProtScout Streamlit Dashboard

This Streamlit application provides an interactive dashboard for exploring PETase enzyme candidates, replicating the functionality of the original React application.

## Features

* Displays candidate PETases identified from OceanDNA metagenomic samples.
* Shows performance metrics (kcat, KM, efficiency, TM) and structural properties.
* Interactive table of top-performing enzymes with pagination.
* Detailed view for selected enzymes, including:
    * Performance metrics and radar chart.
    * Structural properties and structure-function scatter plot.
    * Interactive 3D protein structure viewer using py3Dmol and stmol.
* Interactive 3D visualization of overall property distribution (from pre-generated Plotly HTML).
* Interactive scatter plot showing the Structure-Function Relationship Matrix (TM vs. Efficiency).
* Map showing the geographic origin of samples.

## Project Structure

Ensure your project directory looks like this:

your-project-root/
├── app.py                  # Main Streamlit application script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Directory for data files
│   ├── performance_metrics.tsv
│   └── structural_properties.tsv
├── images/                 # Directory for image assets
│   ├── world_map.png
│   └── protein_visualization_by_kcat_(s^(-1))_high.html # Plotly HTML export
└── pdb-files/              # Directory for PDB files
├── protein_id_1.pdb
├── protein_id_2.pdb
└── ...                 # PDB file for each sequence_id


**Note:**
* The `protein_visualization_by_kcat_(s^(-1))_high.html` file is placed in `images/` for consistency with the React app's path reference, although it's data. Adjust the path in `app.py` if you place it elsewhere.
* PDB files should be named exactly matching the `sequence_id` from your data, followed by `.pdb`, and placed in the `pdb-files/` directory.

## Installation

1.  **Clone the repository or download the files.**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ensure your data files (`.tsv`), image (`.png`), Plotly HTML (`.html`), and PDB files (`.pdb`) are placed in the correct directories (`data/`, `images/`, `pdb-files/`) as shown in the Project Structure.**

## Running the Application

1.  **Navigate to the project's root directory in your terminal.**
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
3.  **The application should open automatically in your web browser.**