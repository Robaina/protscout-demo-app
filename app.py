# fmt: off
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import py3Dmol
import stmol
from pathlib import Path
import math
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re
import geopandas as gpd
import io # Added for image handling
from PIL import Image # Added for image handling

# --- RDKit Import (Add this section) ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
# fmt: on

# --- Custom Title Styling Variables ---
TITLE_COLOR = "#35b779" # A prominent green hex code from the Viridis colormap
H1_FONT_SIZE = "30px"  # Adjust as needed (Default is larger)
H2_FONT_SIZE = "24px"  # Adjust as needed (Default is larger)
H3_FONT_SIZE = "20px"  # Adjust as needed (Default is larger)

# --- Constants ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "images"
PDB_DIR = BASE_DIR / "pdb-files"

PERFORMANCE_METRICS_PATH = DATA_DIR / "performance_metrics.tsv"
STRUCTURAL_PROPERTIES_PATH = DATA_DIR / "structural_properties.tsv"
GEOGRAPHIC_METADATA_PATH = DATA_DIR / "geographic_metadata.tsv"
# LOGO_PATH = IMAGE_DIR / "protscout_logo.png" # Old ProtScout logo path (kept for reference, not used in sidebar)
NAL_LOGO_PATH = IMAGE_DIR / "nal_logo.png" # Path for the New Atlantis Logo
WORLD_SHAPEFILE_PATH = DATA_DIR / "sovereign_states" / "ne_110m_admin_0_sovereignty.shp"

# --- 3D Plot HTML File Mapping ---
HTML_3D_PLOT_FILES = {
    "Catalytic Efficiency (M⁻¹s⁻¹)": "protein_visualization_by_catalytic_efficiency_(M^(-1)s^(-1))_high.html",
    "kcat (s⁻¹)": "protein_visualization_by_kcat_(s^(-1))_high.html",
    "KM (mM)": "protein_visualization_by_KM_(mM)_high.html",
    "Thermal Stability (TM, °C)": "protein_visualization_by_TM_high.html",
    "Optimal Temperature (°C)": "protein_visualization_by_temperature_high.html",
}

# --- Configuration ---
st.set_page_config(
    page_title="ProtScout Dashboard | NAL", # Updated Title slightly
    page_icon="images/nal_logo.png", # Use NAL logo as page icon if desired
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Apply Custom Title Styles ---
st.markdown(f"""
<style>
    /* Target Streamlit's generated header elements */
    h1 {{ /* Corresponds to st.title() */
        color: {TITLE_COLOR} !important;
        font-size: {H1_FONT_SIZE} !important;
    }}
    h2 {{ /* Corresponds to st.header() */
        color: {TITLE_COLOR} !important;
        font-size: {H2_FONT_SIZE} !important;
    }}
    h3 {{ /* Corresponds to st.subheader() */
        color: {TITLE_COLOR} !important;
        font-size: {H3_FONT_SIZE} !important;
    }}
    /* Optional: Style the sidebar title (if it uses h1) */
    /* You might need to inspect the HTML to get the exact selector if this doesn't work */
    .stSidebar h1 {{
         color: {TITLE_COLOR} !important;
         font-size: {H1_FONT_SIZE} !important; /* Or a different size for sidebar */
    }}
    .stSidebar h2 {{
         color: {TITLE_COLOR} !important;
         font-size: {H2_FONT_SIZE} !important;
    }}
    .stSidebar h3 {{
         color: {TITLE_COLOR} !important;
         font-size: {H3_FONT_SIZE} !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def extract_phylum(gtdb_classification: str) -> str:
    """ Extracts the phylum name (p__...) from a GTDB classification string. """
    if not isinstance(gtdb_classification, str): return "Unknown"
    try:
        phylum = re.search(r"p__([^;]+)", gtdb_classification)
        phylum_name = phylum.group(1) if phylum else "Unknown"
        # Handle cases like 'p__' which means unknown at phylum level
        return phylum_name if phylum_name else "Unknown"
    except (AttributeError, TypeError):
        return "Unknown"

# --- Data Loading & Caching ---
@st.cache_data
def load_data():
    """ Loads, merges, cleans, processes data including geo & taxonomic info. """
    try:
        # Load individual data files
        perf_df = pd.read_csv(PERFORMANCE_METRICS_PATH, sep='\t', low_memory=False)
        struct_df = pd.read_csv(STRUCTURAL_PROPERTIES_PATH, sep='\t', low_memory=False)
        geo_df = pd.read_csv(GEOGRAPHIC_METADATA_PATH, sep='\t', low_memory=False)

        # 1. Clean Performance Data (select and rename columns)
        perf_df = perf_df.rename(columns={
            'Substrate': 'substrate', 'SMILES': 'smiles', 'kcat_(s^(-1))': 'kcat',
            'KM_(mM)': 'km', 'solubility': 'solubility', 'pH': 'ph', 'salt': 'salt',
            'temperature': 'temperature', 'TM': 'tm',
            'catalytic_efficiency_(M^(-1)s^(-1))': 'catalytic_efficiency'
        })
        # Select only the columns we intend to use
        perf_df = perf_df[['sequence_id', 'substrate', 'smiles', 'kcat', 'km', 'solubility',
                           'ph', 'salt', 'temperature', 'tm', 'catalytic_efficiency']].copy()
        # Convert numeric columns, coercing errors to NaN
        for col in ['kcat', 'km', 'catalytic_efficiency', 'tm', 'temperature']:
            perf_df[col] = pd.to_numeric(perf_df[col], errors='coerce')

        # 2. Clean Structural Data (select and rename columns)
        struct_df = struct_df.rename(columns={
            'sequence_length': 'sequence_length', 'molecular_weight_kda': 'molecular_weight_kda',
            'isoelectric_point': 'isoelectric_point', 'extinction_coefficient_reduced': 'extinction_coefficient_reduced',
            'extinction_coefficient_oxidized': 'extinction_coefficient_oxidized',
            'instability_index': 'instability_index', 'avg_flexibility': 'avg_flexibility',
            'n_complete': 'n_complete', 'c_complete': 'c_complete'
        })
        struct_df = struct_df[['sequence_id', 'sequence_length', 'molecular_weight_kda', 'isoelectric_point',
                               'extinction_coefficient_reduced', 'extinction_coefficient_oxidized',
                               'instability_index', 'avg_flexibility', 'n_complete', 'c_complete']].copy()
        # Convert numeric columns, coercing errors to NaN
        for col in ['sequence_length', 'molecular_weight_kda', 'isoelectric_point',
                    'extinction_coefficient_reduced', 'extinction_coefficient_oxidized',
                    'instability_index', 'avg_flexibility']:
            struct_df[col] = pd.to_numeric(struct_df[col], errors='coerce')
        # Convert boolean-like columns
        for col in ['n_complete', 'c_complete']:
             struct_df[col] = struct_df[col].astype(bool)

        # 3. Merge Performance and Structural Data on 'sequence_id'
        # Use 'inner' merge to keep only sequences present in both datasets
        merged_df_temp = pd.merge(perf_df, struct_df, on='sequence_id', how='inner')

        # 4. Create 'short_id' in the merged data for linking with geographic data
        # Extracts the part after '___' if present, otherwise keeps the original ID
        def get_short_id(seq_id):
            if isinstance(seq_id, str) and '___' in seq_id:
                return seq_id.split('___', 1)[1]
            return seq_id
        merged_df_temp['short_id'] = merged_df_temp['sequence_id'].apply(get_short_id)

        # 5. Clean Geographic Data & Prepare for Merge
        geo_id_column_name = 'sequence_id' # Define the ID column in the geo file
        # Define desired geographic columns
        geo_cols_to_load = [geo_id_column_name, 'latitude', 'longitude', 'depth',
                            'eez_geoname', 'eez_territory', 'gtdb_classification']

        # Check which columns are actually present in the loaded geo_df
        available_geo_cols = [col for col in geo_cols_to_load if col in geo_df.columns]
        missing_geo_cols = set(geo_cols_to_load) - set(available_geo_cols)
        if missing_geo_cols:
             st.warning(f"⚠️ Columns missing from `{GEOGRAPHIC_METADATA_PATH.name}`: {', '.join(missing_geo_cols)}.")
             # Critical error if the ID column itself is missing
             if geo_id_column_name not in available_geo_cols:
                 st.error(f"❌ Critical Error: Column '{geo_id_column_name}' not found in geographic data file. Cannot merge.")
                 return None # Stop data loading

        # Select only available columns and rename the ID column to 'short_id' for merging
        geo_df = geo_df[available_geo_cols].copy()
        geo_df = geo_df.rename(columns={geo_id_column_name: 'short_id'})

        # Convert numeric geo columns, coercing errors
        for col in ['latitude', 'longitude', 'depth']:
            if col in geo_df.columns:
                geo_df[col] = pd.to_numeric(geo_df[col], errors='coerce')

        # --- Final Merge using 'short_id' ---
        # Use 'inner' merge to keep only sequences with matching geographic info
        merged_df = pd.merge(merged_df_temp, geo_df, on='short_id', how='inner')

        # Check if the final merge resulted in an empty DataFrame
        if merged_df.empty:
            st.warning("⚠️ Warning: No matching entries found after merging performance/structural data with geographic data based on the short ID.")
            return None # Return None if merge fails

        # --- Post-Merge Data Processing ---
        # Extract Phylum from GTDB classification if available
        if 'gtdb_classification' in merged_df.columns:
            merged_df['phylum'] = merged_df['gtdb_classification'].apply(extract_phylum)
        else:
            merged_df['phylum'] = "Unknown"
            # Warn if the column wasn't even available in the geo file
            if 'gtdb_classification' not in available_geo_cols:
                 st.warning("⚠️ 'gtdb_classification' column not found in geographic data. Taxonomic plot will show 'Unknown'.")

        # Drop rows with NaN values in crucial columns needed for analysis/plotting
        crucial_cols = ['catalytic_efficiency', 'kcat', 'km', 'tm', 'instability_index', 'temperature',
                        'latitude', 'longitude'] # Define essential columns
        # Check which of these crucial columns actually exist in the merged_df
        cols_to_check_na = [col for col in crucial_cols if col in merged_df.columns]
        merged_df.dropna(subset=cols_to_check_na, inplace=True)

        # Recalculate derived values after potential row drops/data cleaning
        # Inverted KM (normalized for radar plot, higher is better)
        merged_df['normalized_km_inv'] = merged_df['km'].apply(lambda x: 1 / x * 2 if pd.notna(x) and x != 0 else 0) # Scale factor 2 is arbitrary
        # Normalized efficiency (e.g., per 1000 for radar plot scale)
        merged_df['normalized_efficiency'] = merged_df['catalytic_efficiency'] / 1000 # Example scaling

        # Sort by catalytic efficiency (descending) as the primary ranking metric
        processed_df = merged_df.sort_values(by='catalytic_efficiency', ascending=False).reset_index(drop=True)

        # Fill remaining numeric NaNs with 0 (or another appropriate strategy if needed)
        numeric_cols_to_fill = ['instability_index', 'tm', 'catalytic_efficiency', 'kcat', 'km', 'temperature',
                                'depth', 'latitude', 'longitude',
                                'sequence_length', 'molecular_weight_kda', 'isoelectric_point',
                                'extinction_coefficient_reduced', 'extinction_coefficient_oxidized', 'avg_flexibility']
        for col in numeric_cols_to_fill:
             if col in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[col]):
                 # Fill only if the column exists and is numeric
                 processed_df[col] = processed_df[col].fillna(0) # Replace NaN with 0

        return processed_df

    except FileNotFoundError as e:
        st.error(f"❌ Error: Required data file not found: `{e.filename}`.")
        st.info(f"Ensure data files (`.tsv`) are in `{DATA_DIR}` (or subdirs), shapefiles in `{DATA_DIR}/sovereign_states/`, and `app.py` is in the parent directory.")
        return None
    except KeyError as e:
        st.error(f"❌ Error: Expected column not found during data processing: `{e}`. Check column names in input files and code.")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during data loading: {e}")
        st.exception(e) # Provides traceback in the terminal/logs
        return None

# --- Load Data ---
df = load_data()

# --- Initialize Session State ---
# Used to keep track of user selections and pagination across reruns
if 'selected_enzyme_id' not in st.session_state:
    st.session_state.selected_enzyme_id = None # Stores the sequence_id of the currently selected enzyme
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0 # Page number for the enzyme table
if 'items_per_page' not in st.session_state:
    st.session_state.items_per_page = 10 # Default number of items per page

# --- Helper Functions ---
# (extract_phylum is defined before load_data)

def get_radar_data(enzyme):
    """ Prepares data for the performance radar chart for a selected enzyme. """
    if enzyme is None:
        return pd.DataFrame() # Return empty DataFrame if no enzyme data

    # Define metrics and corresponding values from the enzyme dictionary
    data = {
        'Metric': ['Catalytic Efficiency', 'kcat', 'KM (inverted)', 'Thermal Stability', 'Operating Temperature'],
        'Value': [
            enzyme.get('normalized_efficiency', 0), # Use pre-calculated normalized value
            enzyme.get('kcat', 0),
            enzyme.get('normalized_km_inv', 0),    # Use pre-calculated normalized inverted KM
            enzyme.get('tm', 0),
            enzyme.get('temperature', 0)
        ],
        'theta': ['Catalytic Eff.', 'kcat', '1/KM', 'TM', 'Temp. (°C)'] # Short labels for radar axes
    }
    return pd.DataFrame(data)

def format_value(value, precision=2):
    """ Formats numeric values for display, handling NA and large numbers. """
    if pd.isna(value):
        return "N/A"
    try:
        if isinstance(value, (float, int)):
             # Use scientific notation for large absolute values if precision is requested
             if abs(value) > 1e5 and precision > 0 :
                 # Exception: Don't use scientific for these specific integer columns
                 # (This check is slightly redundant now as explicit formatting is applied later, but kept for robustness)
                 if isinstance(value, int) and value > 1e5 and precision == 0:
                      return f"{int(value)}"
                 return f"{value:.{precision}e}"
             # Format as integer if it's a whole number or precision is 0
             if precision == 0 or (isinstance(value, int) or value == int(value)):
                 return f"{int(value)}"
             # Otherwise, format as float with specified precision
             return f"{value:.{precision}f}"
        if isinstance(value, bool):
            # Format boolean values clearly
            return "Complete" if value else "Incomplete"
        # Return other types as string
        return str(value)
    except (TypeError, ValueError):
        # Fallback for any unexpected types
        return str(value)

def display_3d_structure(pdb_file_path: Path, sequence_length: int, style: str = 'cartoon', color: str = 'position_viridis'):
    """ Renders the 3D protein structure using stmol and py3Dmol. """
    try:
        # Read PDB file content
        with open(pdb_file_path, 'r') as f:
            pdb_data = f.read()

        # Initialize py3Dmol view
        view = py3Dmol.view()
        view.addModel(pdb_data, 'pdb')

        # Define style and color options
        mol_color_option = color
        mol_style_dict = {
            'cartoon': {'cartoon': {}},
            'ballAndStick': {'stick': {'radius': 0.15}, 'sphere': {'scale': 0.2}},
            'ribbon': {'ribbon': {}},
            'lines': {'line': {}}
        }
        mol_style = mol_style_dict.get(style, {'cartoon': {}}) # Default to cartoon
        style_key = list(mol_style.keys())[0] # Get the primary style key ('cartoon', 'stick', etc.)

        # Apply coloring scheme based on selection
        if mol_color_option == 'spectrum':
            mol_style[style_key]['colorscheme'] = 'spectrum' # Rainbow coloring
        elif mol_color_option == 'chainHetatm':
            mol_style[style_key]['colorscheme'] = 'chain' # Color by chain ID
        elif mol_color_option == 'whiteCarbon':
            mol_style[style_key]['colorscheme'] = 'elem' # Color by element (CPK colors)
        elif mol_color_option == 'ssPyMOL':
            # Special case for PyMOL secondary structure coloring
            view.setStyle({'cartoon': {'colorscheme': 'ssPyMOL'}})
            mol_style = None # Don't apply the general style on top
        elif mol_color_option == 'residue':
            mol_style[style_key]['colorscheme'] = 'chain' # Often similar to chain for single-chain proteins
        elif mol_color_option == 'position_viridis':
            # Custom gradient coloring along the sequence length using Viridis
            if sequence_length > 0:
                viridis = cm.get_cmap('viridis', sequence_length)
                hex_colors = [mcolors.to_hex(viridis(i)) for i in range(sequence_length)]
                custom_colorscheme = {
                    'prop': 'resi', 'gradient': 'linear', 'colors': hex_colors,
                    'min': 1, 'max': sequence_length # Map residue index (1-based) to color
                }
                mol_style[style_key]['colorscheme'] = custom_colorscheme
            else:
                # Fallback if sequence length is invalid
                st.warning("Sequence length unavailable for Viridis coloring. Using spectrum.")
                mol_style[style_key]['colorscheme'] = 'spectrum'
        else:
             # Default fallback coloring
             mol_style[style_key]['colorscheme'] = 'spectrum'

        # Apply the determined style if not handled specially (like ssPyMOL)
        if mol_style:
            view.setStyle({}, mol_style)

        # Set background and zoom
        view.setBackgroundColor('#1A1A1A') # Dark background
        view.zoomTo() # Auto-zoom to fit the structure

        # Render using stmol component - REMOVED KEY ARGUMENT
        stmol.showmol(view, width=800, height=350)

    except FileNotFoundError:
        st.error(f"❌ PDB file not found: `{pdb_file_path.name}`. Ensure PDB files are in the `{PDB_DIR}` directory.")
    except Exception as e:
        st.error(f"❌ Error rendering 3D structure for {pdb_file_path.name}: {e}")
        st.exception(e) # Log the full traceback

# --- Generate substrate image ---
@st.cache_data(show_spinner=False)
def render_smiles(smiles: str):
    """ Renders a SMILES string to a PNG image using RDKit. """
    # Check if RDKit is available and SMILES is valid
    if not RDKIT_AVAILABLE:
        return None, "RDKit library not installed. Cannot render structure."
    if not isinstance(smiles, str) or not smiles:
        return None, "No valid SMILES string provided."

    try:
        # Create RDKit molecule object from SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: # Check if MolFromSmiles failed (invalid SMILES)
            return None, f"Invalid SMILES string: `{smiles}`"

        # Generate image using RDKit Draw
        img = Draw.MolToImage(mol, size=(300, 300), kekulize=True) # Adjust size as needed

        # Convert PIL image to bytes (PNG format)
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        return bio.getvalue(), None # Return image bytes and no error message

    except Exception as e:
        # Catch any other RDKit errors during rendering
        return None, f"Error generating structure from SMILES `{smiles}`: {e}"


@st.cache_data(show_spinner=False)
def create_geo_map_with_highlights(data_df: pd.DataFrame) -> go.Figure:
    """ Generates a world map showing sample locations and highlighting countries with samples. """
    # Check for required columns
    if 'latitude' not in data_df.columns or 'longitude' not in data_df.columns:
        st.warning("⚠️ Latitude/Longitude columns missing. Cannot generate map.")
        return go.Figure().update_layout(title="Geographic data unavailable")
    if 'eez_territory' not in data_df.columns:
        st.warning("⚠️ 'eez_territory' column missing. Cannot highlight countries.")
        # Still proceed to plot points if lat/lon exist
        pass # Allow plotting points even without territory info

    # Check if shapefile exists
    if not WORLD_SHAPEFILE_PATH.is_file():
        st.error(f"❌ Shapefile not found: `{WORLD_SHAPEFILE_PATH}`. Cannot draw country boundaries.")
        return go.Figure().update_layout(title="Shapefile missing")

    # Prepare data for plotting (drop NA lat/lon)
    plot_df = data_df.dropna(subset=['latitude', 'longitude']).copy()

    # --- Load Shapefile and Prepare Highlighting ---
    try:
        world = gpd.read_file(WORLD_SHAPEFILE_PATH)
    except Exception as e:
        st.error(f"❌ Error loading shapefile: {e}")
        return go.Figure().update_layout(title="Shapefile loading error")

    # Identify countries with samples IF eez_territory exists
    countries_with_samples_mapped = set()
    if 'eez_territory' in plot_df.columns:
        plot_df_territory = plot_df.dropna(subset=['eez_territory'])
        # Filter out ambiguous/non-country territories
        plot_df_territory = plot_df_territory[~plot_df_territory['eez_territory'].str.contains("Unknown|Intl.|Joint", case=False, na=False)]
        if not plot_df_territory.empty:
            countries_with_samples = set(plot_df_territory['eez_territory'].unique())
            # Map names if needed (e.g., "United States" in data vs "United States of America" in shapefile)
            country_mapping = {"United States": "United States of America",}
            countries_with_samples_mapped = {country_mapping.get(c, c) for c in countries_with_samples}

    # Define the country name column in the shapefile (VERIFY THIS!)
    shapefile_country_col = 'NAME'
    if shapefile_country_col not in world.columns:
        st.error(f"❌ Column '{shapefile_country_col}' not found in shapefile. Cannot match countries.")
        st.info(f"Available columns in shapefile: {', '.join(world.columns)}")
        # Proceed without highlighting if column is missing
        world['highlight_cat'] = 'No Samples' # Set all to default color
    else:
        # Create categorical column for coloring based on mapped country names
        world['highlight_cat'] = world[shapefile_country_col].apply(
            lambda name: 'Has Samples' if name in countries_with_samples_mapped else 'No Samples'
        )

    # --- Define Colors ---
    highlight_color = '#FDE74C' # Bright Yellow highlight (Viridis-like)
    default_color = '#404040'   # Darker Gray default land
    marker_color = '#7AD151'    # Bright Green points (Viridis-like)
    marker_line_color = '#FFFFFF' # White outline for markers
    dark_ocean_color = '#1A1A1A'  # Very dark background/ocean

    # --- Create Choropleth Base Map ---
    fig = px.choropleth(
        world, geojson=world.geometry, locations=world.index, # Use index for location matching
        color='highlight_cat', # Use the categorical column for coloring
        color_discrete_map={'Has Samples': highlight_color, # Map categories to colors
                            'No Samples': default_color},
        projection="natural earth", # Choose a map projection
    )
    # Make the base map subtle
    fig.update_traces(marker_line_width=0.5, marker_line_color='#555555', marker_opacity=0.7, selector=dict(type='choropleth'))

    # --- Add Scatter Points for Samples ---
    # Define columns for hover tooltip
    hover_cols = {'latitude': ':.2f', 'longitude': ':.2f', 'short_id': True,
                  'catalytic_efficiency': ':.1e', 'tm': ':.1f', 'depth': ':.1f',
                  'eez_territory': True, 'phylum': True}
    # Filter hover columns to only those present in the dataframe
    hover_cols = {k: v for k, v in hover_cols.items() if k in plot_df.columns}
    # Prepare custom data for hover template
    custom_data_cols = [col for col in hover_cols.keys() if col in plot_df.columns and col not in ['latitude', 'longitude']]
    customdata_df = plot_df[custom_data_cols]

    # Build hover template dynamically
    ht = "<b>%{customdata[0]}</b><br>" # Start with short_id (assuming it's first in custom_data_cols)
    ht += "Lat: %{lat:.2f}, Lon: %{lon:.2f}<br>"
    col_index = 1 # Start from the second custom data column
    for col in custom_data_cols[1:]:
        label = col.replace('_', ' ').title() # Format label nicely
        unit = " °C" if col == 'tm' else " m" if col == 'depth' else "" # Add units
        format_code = hover_cols.get(col, '') # Get format code if specified
        format_str = f":{format_code}" if isinstance(format_code, str) and format_code else "" # Construct format string
        ht += f"{label}: %{{customdata[{col_index}]{format_str}}}{unit}<br>" # Add line to template
        col_index += 1
    ht += "<extra></extra>" # Remove extra Plotly info box

    # Add the scatter points trace
    fig.add_trace(
        go.Scattergeo(
            lon = plot_df['longitude'],
            lat = plot_df['latitude'],
            mode = 'markers',
            marker = dict(
                color=marker_color,
                opacity=0.8,
                size=10,
                line=dict(width=0.5, color=marker_line_color) # Add outline to markers
            ),
            hoverinfo = 'text', # Use the custom hover template
            customdata = customdata_df, # Provide the custom data
            hovertemplate = ht, # Assign the template string
            name="Sample Locations" # Name for legend (optional)
        )
    )

    # --- Update Layout for Dark Theme and Appearance ---
    fig.update_layout(
        title=None, # Remove default title
        showlegend=False, # Hide legend
        margin={"r":5,"t":5,"l":5,"b":5}, # Minimize margins
        paper_bgcolor=dark_ocean_color, # Background of the entire figure area
        plot_bgcolor=dark_ocean_color, # Background of the plotting area
        geo = dict(
            bgcolor=dark_ocean_color, # Map background
            oceancolor=dark_ocean_color, # Ocean color
            landcolor=default_color, # Default land color (used where not highlighted)
            showland=True, showocean=True,
            showcountries=False, # Hide default country borders (we use choropleth)
            showsubunits=False,
            lataxis_showgrid=False, lonaxis_showgrid=False # Hide grid lines
        )
    )
    # Ensure the underlying geo map features are not visible if using choropleth for land
    fig.update_geos(visible=False)

    return fig

# Donut Chart Function
def create_donut_chart(data_series: pd.Series, title: str, center_text: str = "", text_color: str = '#E0E0E0') -> go.Figure:
    """ Creates a Plotly Express donut chart with custom styling. """
    if data_series.empty or data_series.sum() == 0: # Handle empty or all-zero data
        fig = go.Figure()
        fig.update_layout(
            title={'text': f"{title}<br><i>(No data available)</i>", 'x': 0.5, 'xanchor': 'center', 'font_color': text_color},
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[{'text': 'N/A', 'x': 0.5, 'y': 0.5, 'font_size': 20, 'showarrow': False, 'font_color': text_color}]
        )
        return fig

    # Create the basic donut chart
    fig = px.pie(
        values=data_series.values,
        names=data_series.index,
        # title=title, # Title added in update_layout for better control
        hole=0.6, # Size of the center hole
        color_discrete_sequence=px.colors.sequential.Viridis # Color scheme
    )

    # Update trace appearance
    fig.update_traces(
        textposition='outside', # Place labels outside slices
        textinfo='percent+label', # Show percentage and label
        marker=dict(line=dict(color='#1A1A1A', width=2)), # Add lines between slices
        pull=[0.02] * len(data_series), # Slightly pull slices apart
        insidetextorientation='radial' # Orientation if text were inside
    )

    # Update layout for appearance and title
    fig.update_layout(
        height=400, # Set chart height
        showlegend=False, # Hide legend
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)', # Transparent plot area
        font_color=text_color, # Set default font color
        title={'text': title, 'x': 0.5, 'xanchor': 'center', 'font_color': text_color}, # Center title
        margin=dict(t=60, b=20, l=20, r=20), # Adjust margins
        # Add center text annotation
        annotations=[dict(text=center_text, x=0.5, y=0.5, font_size=20, showarrow=False, font_color=text_color)]
    )
    return fig

# --- Sidebar Content ---

# Display NAL Logo at the top
if NAL_LOGO_PATH.is_file():
    st.sidebar.image(str(NAL_LOGO_PATH), width=150) # Adjust width as needed
else:
    st.sidebar.warning(f"NAL Logo not found at {NAL_LOGO_PATH}")

# Add the Tool Title
st.sidebar.title("ProtScout")
st.sidebar.markdown("---") # Add a divider

# Add Navigation Section
st.sidebar.subheader("Dashboard Navigation") # Changed from title to subheader
page = st.sidebar.radio(
    "Select View:",
    ("Enzyme Explorer", "Property Visualizations", "Geographic Distribution"),
    key="navigation_radio",
    label_visibility="collapsed" # Hide the label "Select View:"
)
st.sidebar.markdown("---") # Add a divider

# Add Expandable Section for Detailed Text
with st.sidebar.expander("About ProtScout & This Demo", expanded=False): # Set expanded=False
    st.markdown("""
    Welcome! This dashboard demonstrates **ProtScout**, NAL's proprietary AI toolkit for rapid *in silico* functional characterization and ranking of protein sequences. ProtScout accelerates the discovery of valuable proteins by predicting their potential before costly lab experiments.
    """)

    st.markdown("""
    **Case Study: Novel PETase Discovery**

    Here, ProtScout analyzes potential PET-degrading enzymes (PETases) identified from marine metagenomic samples (**OceanDNA**). These novel sequences are compared against known, experimentally validated PETases from the **PAZy database**.

    * **Goal:** Identify promising, previously unknown enzymes capable of breaking down PET plastic.
    * **Source:** Diverse marine environments, offering unique biological potential.
    * **Method:** AI-driven ranking based on predicted functional characteristics.
    """)

    st.markdown("""
    **Impact & Opportunity:**

    Discovering effective PETases is crucial for tackling plastic pollution. ProtScout significantly speeds up this search, pinpointing high-potential candidates from vast datasets. This capability extends to discovering enzymes and proteins for various industrial, environmental, and therapeutic applications.
    """)

    st.markdown("""
    **Leveraging ProtScout Insights:**

    ProtScout serves as a powerful discovery engine. The high-potential candidates identified can then be channeled into further development:
    1.  Direct prioritization for efficient laboratory validation to confirm predicted functions experimentally.
    2.  Selection for protein engineering efforts aimed at optimizing key characteristics such as stability, activity, or substrate specificity.
    """)

# --- Main App Layout ---
if df is not None: # Only proceed if data loading was successful

    # --- Page 1: Enzyme Explorer ---
    if page == "Enzyme Explorer":
        st.header("🚀 ProtScout: Enzyme Explorer")
        st.markdown("""
        Browse PETase candidates discovered and ranked by ProtScout, primarily based on predicted catalytic efficiency.
        The table shows key performance and structural properties, plus origins. Higher ranks indicate greater predicted potential.

        **Select a row in the table below to view the detailed profile for that specific enzyme.**
        """)

        # --- Pagination Logic ---
        total_items = len(df)
        items_per_page = st.session_state.items_per_page
        total_pages = math.ceil(total_items / items_per_page)
        # Ensure current_page is valid
        current_page = max(0, min(st.session_state.current_page, total_pages - 1))
        st.session_state.current_page = current_page # Update state if corrected

        start_idx = current_page * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        paginated_df = df.iloc[start_idx:end_idx]

        # --- Pagination Controls ---
        cols_page = st.columns([3, 1, 1, 1]) # Adjust column ratios as needed
        with cols_page[0]:
            st.markdown(f"Displaying **{start_idx + 1}** - **{end_idx}** of **{total_items}** enzymes")
        with cols_page[1]:
            if st.button("⬅️ Previous", use_container_width=True, disabled=(current_page == 0), key="prev_page"):
                st.session_state.current_page -= 1
                st.session_state.selected_enzyme_id = None # Reset selection on page change
                st.rerun()
        with cols_page[2]:
            if st.button("Next ➡️", use_container_width=True, disabled=(current_page >= total_pages - 1), key="next_page"):
                st.session_state.current_page += 1
                st.session_state.selected_enzyme_id = None # Reset selection on page change
                st.rerun()
        with cols_page[3]:
            options = [5, 10, 20, 50, 100]
            # Set default index robustly
            try: index = options.index(items_per_page)
            except ValueError: index = 1 # Default to 10 if current value isn't in options
            new_ipp = st.selectbox("Items/page", options=options, index=index, key='items_per_page_selector', label_visibility="collapsed")
            if new_ipp != items_per_page:
                st.session_state.items_per_page = new_ipp
                st.session_state.current_page = 0 # Go to first page on changing items per page
                st.session_state.selected_enzyme_id = None
                st.rerun()

        # --- DataFrame Display --- # MODIFIED SECTION STARTS HERE
        # Define columns to display: Core + ALL Structural Properties
        display_cols = [
            'short_id', 'catalytic_efficiency', 'kcat', 'km', 'temperature', 'tm', # Core performance
            'sequence_length', 'molecular_weight_kda', 'isoelectric_point', # Basic structural
            'instability_index', 'avg_flexibility', # Stability/Flexibility
            'extinction_coefficient_reduced', 'extinction_coefficient_oxidized', # Other structural
            'n_complete', 'c_complete' # Completeness flags
        ]
        # Conditionally add geographic column if available
        if 'eez_territory' in df.columns: display_cols.append('eez_territory')
        # Ensure all selected columns actually exist in the dataframe
        display_cols = [col for col in display_cols if col in paginated_df.columns]

        # Create the dataframe slice with only the columns we want
        formatted_paginated_df = paginated_df[display_cols].copy()

        # Define formatting rules based on column type/content
        cols_to_format_numeric = {
            'catalytic_efficiency': ':.1e', # Scientific notation
            'kcat': ':.2f',
            'km': ':.2f',
            'temperature': ':.1f',
            'tm': ':.1f',
            'molecular_weight_kda': ':.2f',
            'isoelectric_point': ':.2f',
            'instability_index': ':.2f',
            'avg_flexibility': ':.3f',
            'sequence_length': ':.0f', # Integer
            'extinction_coefficient_reduced': ':.0f', # Integer
            'extinction_coefficient_oxidized': ':.0f' # Integer
        }
        cols_to_format_boolean = ['n_complete', 'c_complete']
        cols_to_format_text = ['eez_territory'] # Handle NAs for text/categorical

        # Apply formatting
        for col, fmt in cols_to_format_numeric.items():
            if col in formatted_paginated_df.columns:
                if col == 'catalytic_efficiency':
                    # Direct scientific formatting for catalytic efficiency, handling NA
                    formatted_paginated_df[col] = formatted_paginated_df[col].apply(
                        lambda x: f"{x:.1e}" if pd.notna(x) else "N/A"
                    )
                else:
                    # Use format_value for other numeric columns
                    precision = 0 # Default precision (e.g., for integers)
                    if 'f' in fmt: # Check if float format is specified
                        try:
                            precision = int(fmt.split('.')[-1].replace('f', ''))
                        except ValueError:
                            precision = 2 # Fallback precision for floats
                    # format_value handles integers (precision=0), large numbers (e), NAs itself
                    formatted_paginated_df[col] = formatted_paginated_df[col].apply(
                        lambda x: format_value(x, precision)
                    )

        # Boolean and Text formatting remains the same
        for col in cols_to_format_boolean:
            if col in formatted_paginated_df.columns:
                formatted_paginated_df[col] = formatted_paginated_df[col].apply(lambda x: format_value(x)) # format_value handles bools

        for col in cols_to_format_text:
             if col in formatted_paginated_df.columns:
                 formatted_paginated_df[col] = formatted_paginated_df[col].fillna("N/A")

        # Rename columns for better readability in the table header
        rename_map = {
            'short_id': 'ID',
            'catalytic_efficiency': 'Catalytic Eff.',
            'kcat': 'kcat (s⁻¹)',
            'km': 'KM (mM)',
            'temperature': 'Opt. Temp (°C)', # Shortened
            'tm': 'TM (°C)',
            'sequence_length': 'Length (aa)',
            'molecular_weight_kda': 'Mol. Wt (kDa)',
            'isoelectric_point': 'pI',
            'instability_index': 'Instability Idx',
            'avg_flexibility': 'Avg. Flex.',
            'extinction_coefficient_reduced': 'Ext. Coeff. (Red)',
            'extinction_coefficient_oxidized': 'Ext. Coeff. (Oxi)',
            'n_complete': 'N-Term', # Shortened
            'c_complete': 'C-Term', # Shortened
            'eez_territory': 'Origin Territory'
        }
        # Apply renaming only for columns that are actually present
        formatted_paginated_df = formatted_paginated_df.rename(
            columns={k: v for k, v in rename_map.items() if k in formatted_paginated_df.columns}
        )

        # Display the dataframe with selection enabled
        st.dataframe(
            formatted_paginated_df,
            use_container_width=True,
            hide_index=True,
            key="enzyme_table", # Key to access selection state
            on_select="rerun", # Trigger a rerun when selection changes
            selection_mode="single-row" # Allow selecting only one row
        )
        # MODIFIED SECTION ENDS HERE

        # --- Selection Logic ---
        user_selection = st.session_state.get("enzyme_table", {}).get("selection", {"rows": []})
        user_selected_index = user_selection["rows"][0] if user_selection["rows"] else None

        enzyme_to_display_id = None # Initialize ID to display

        if user_selected_index is not None:
            # If user explicitly selected a row
            try:
                # Get the sequence_id from the paginated dataframe based on the selected index
                enzyme_to_display_id = paginated_df.iloc[user_selected_index]['sequence_id']
                # Update session state only if the selection has changed
                if st.session_state.selected_enzyme_id != enzyme_to_display_id:
                    st.session_state.selected_enzyme_id = enzyme_to_display_id
                    # No rerun needed here, the rerun was triggered by on_select="rerun"
            except IndexError:
                # Handle case where index might be out of bounds (shouldn't happen often)
                st.warning("Could not retrieve selected enzyme from the current page.")
                st.session_state.selected_enzyme_id = None
                enzyme_to_display_id = None
        elif st.session_state.selected_enzyme_id is not None:
            # If a selection exists in session state (e.g., from previous interaction or default)
            enzyme_to_display_id = st.session_state.selected_enzyme_id
            # Check if the selected ID is actually visible on the current page
            if enzyme_to_display_id not in paginated_df['sequence_id'].values:
                # If not visible (e.g., user paged away), clear the selection state to avoid showing wrong details
                # st.session_state.selected_enzyme_id = None # Option 1: Clear selection
                # enzyme_to_display_id = None
                pass # Option 2: Keep the ID but don't display details (handled later)
        else:
            # If no selection exists (first load or after clearing)
            # Default to the first item on the current page if the page is not empty
            if not paginated_df.empty:
                enzyme_to_display_id = paginated_df.iloc[0]['sequence_id']
                st.session_state.selected_enzyme_id = enzyme_to_display_id
                # Use a flag to trigger rerun ONLY for the initial default selection setup
                if 'initial_default_set' not in st.session_state:
                    st.session_state.initial_default_set = True
                    st.rerun() # Rerun to make the default selection visually highlighted

        # Clean up the initial default set flag after the potential rerun
        if 'initial_default_set' in st.session_state and st.session_state.selected_enzyme_id is not None:
             del st.session_state['initial_default_set']


        # --- Fetch Data for the Selected Enzyme ---
        selected_enzyme_data = None
        if enzyme_to_display_id:
            # Find the full data row in the original (unpaginated) dataframe
            selected_enzyme_row = df.loc[df['sequence_id'] == enzyme_to_display_id]
            if not selected_enzyme_row.empty:
                selected_enzyme_data = selected_enzyme_row.iloc[0].to_dict()
            else:
                # This might happen if the ID got stale or data changed unexpectedly
                st.warning(f"Selected enzyme ID '{enzyme_to_display_id}' not found in the main dataset.")
                st.session_state.selected_enzyme_id = None # Clear the invalid ID


        # --- Enzyme Details Section ---
        st.divider()
        if selected_enzyme_data:
            st.subheader(f"🔬 Enzyme Profile: `{selected_enzyme_data.get('short_id', 'N/A')}`")
            st.markdown("Detailed properties and visualizations for the selected enzyme.")

            col_details_left, col_details_right = st.columns([0.55, 0.45]) # Adjust ratio if needed

            with col_details_left:
                # --- Combined Performance & Conditions Table ---
                st.markdown("##### Performance & Operating Conditions")
                # Prepare data for the table
                perf_cond_metrics = {
                    "Catalytic Eff. (M⁻¹s⁻¹)": f"{selected_enzyme_data.get('catalytic_efficiency'):.1e}" if pd.notna(selected_enzyme_data.get('catalytic_efficiency')) else "N/A",
                    "kcat (s⁻¹)": format_value(selected_enzyme_data.get('kcat'), 2),
                    "KM (mM)": format_value(selected_enzyme_data.get('km'), 2),
                    "Thermal Stability TM (°C)": format_value(selected_enzyme_data.get('tm'), 1),
                    "Optimal Temp (°C)": format_value(selected_enzyme_data.get('temperature'), 1),
                    "Optimal pH": format_value(selected_enzyme_data.get('ph'), 1),
                    "Salt Condition": selected_enzyme_data.get('salt', 'N/A'),
                    "Solubility": selected_enzyme_data.get('solubility', 'N/A'),
                }
                # Add geographic/taxonomic info if available
                if 'depth' in selected_enzyme_data: perf_cond_metrics["Source Depth (m)"] = format_value(selected_enzyme_data.get('depth'), 1)
                if 'eez_territory' in selected_enzyme_data: perf_cond_metrics["Source Territory"] = selected_enzyme_data.get('eez_territory', 'N/A')
                if 'phylum' in selected_enzyme_data: perf_cond_metrics["Source Phylum"] = selected_enzyme_data.get('phylum', 'N/A')

                perf_cond_df = pd.DataFrame(perf_cond_metrics.items(), columns=["Metric", "Value"])
                # Display as a simple table using st.dataframe
                st.dataframe(perf_cond_df, hide_index=True, use_container_width=True)

                # --- Combined Structural & Stability Table ---
                st.markdown("##### Structural Properties & Stability")
                instability_idx = selected_enzyme_data.get('instability_index')
                # Add a simple stability indicator based on the index
                stability_label = " (Stable)" if instability_idx is not None and instability_idx < 40 else " (Potentially Unstable)" if instability_idx is not None else ""
                struct_stab_metrics = {
                    "Sequence Length (aa)": format_value(selected_enzyme_data.get('sequence_length'), 0),
                    "Molecular Weight (kDa)": format_value(selected_enzyme_data.get('molecular_weight_kda'), 2),
                    "Isoelectric Point (pI)": format_value(selected_enzyme_data.get('isoelectric_point'), 2),
                    "Avg. Flexibility": format_value(selected_enzyme_data.get('avg_flexibility'), 3),
                    "Instability Index": f"{format_value(instability_idx, 2)}{stability_label}",
                    "Ext. Coeff. (Red)": format_value(selected_enzyme_data.get('extinction_coefficient_reduced'), 0), # M-1 cm-1
                    "Ext. Coeff. (Oxi)": format_value(selected_enzyme_data.get('extinction_coefficient_oxidized'), 0), # M-1 cm-1
                    "N-Terminus Complete": format_value(selected_enzyme_data.get('n_complete')),
                    "C-Terminus Complete": format_value(selected_enzyme_data.get('c_complete')),
                }
                struct_stab_df = pd.DataFrame(struct_stab_metrics.items(), columns=["Property", "Value"])
                st.dataframe(struct_stab_df, hide_index=True, use_container_width=True)


                # --- Substrate Section ---
                st.markdown("##### Substrate Information")
                substrate_name = selected_enzyme_data.get('substrate', 'N/A')
                smiles_string = selected_enzyme_data.get('smiles') # Get raw SMILES

                st.markdown(f"**Substrate Name:** {substrate_name}")

                if smiles_string and isinstance(smiles_string, str):
                    # Attempt to render the SMILES string
                    img_bytes, error_msg = render_smiles(smiles_string)
                    if img_bytes:
                        # If rendering successful, display the image
                        st.image(img_bytes, caption=f"Chemical structure of {substrate_name}", use_container_width=False, width=250) # Adjust width
                        st.caption(f"SMILES Notation: `{smiles_string}`") # Show SMILES below image
                    elif error_msg:
                        # If rendering failed, show error and SMILES string
                        st.warning(f"⚠️ Could not display structure: {error_msg}")
                        st.text(f"SMILES Notation: {smiles_string}") # Fallback to text
                    # Implicitly handles case where RDKit might be unavailable (error_msg will indicate this)
                elif smiles_string:
                    # Handles case where SMILES is present but not a string (e.g., NaN converted unexpectedly)
                    st.info("Substrate SMILES available but not in a valid format.")
                    st.text(f"SMILES data: {smiles_string}")
                else:
                    # If no SMILES string is available
                    st.info("No chemical structure (SMILES) data available for this substrate.")


            with col_details_right:
                # --- 3D Structure ---
                st.markdown("**🧊 Predicted 3D Structure (AlphaFold)**") # Assuming AlphaFold prediction
                # Construct the expected PDB file path
                pdb_file = PDB_DIR / f"{selected_enzyme_data.get('sequence_id', '')}.pdb"
                seq_len = selected_enzyme_data.get('sequence_length', 0) # Needed for Viridis coloring

                if pdb_file.is_file():
                    # Add controls for style and color
                    ctrl_col1, ctrl_col2 = st.columns(2)
                    with ctrl_col1:
                        view_style = st.selectbox("Style", ['cartoon', 'ballAndStick', 'lines', 'ribbon'], index=0, key='style_3d', label_visibility="collapsed")
                    with ctrl_col2:
                        options = ['position_viridis', 'chainHetatm', 'spectrum', 'ssPyMOL', 'whiteCarbon']
                        labels = {
                            'position_viridis': 'Position (Viridis)', 'chainHetatm': 'By Chain',
                            'spectrum': 'Position (Rainbow)', 'ssPyMOL': 'Sec. Structure',
                            'whiteCarbon': 'By Element', 'residue': 'By Residue (Chain)'
                        }
                        selected_color = st.selectbox(
                            "Color", options, index=options.index('position_viridis'),
                            format_func=lambda x: labels.get(x, x), key='color_3d', label_visibility="collapsed"
                            )
                    # Display the structure
                    display_3d_structure(pdb_file, seq_len, view_style, selected_color)
                else:
                    st.warning(f"⚠️ Predicted structure PDB file not found: `{pdb_file.name}`")

                # --- Radar Chart ---
                st.markdown("**🎯 Key Performance Profile**")
                st.caption("Normalized comparison of key metrics (higher values outward are generally better).")
                radar_df = get_radar_data(selected_enzyme_data)
                if not radar_df.empty:
                    fig_radar = px.line_polar(radar_df, r='Value', theta='theta', line_close=True, markers=True)
                    # Define colors for dark theme
                    line_color = '#FDE74C' # Viridis Yellow
                    fill_color_transparent = 'rgba(253, 231, 76, 0.3)' # Semi-transparent Yellow fill
                    light_text_color = '#E0E0E0'
                    grid_color = '#555555'

                    # Update layout for dark theme and appearance
                    fig_radar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
                        polar_bgcolor='rgba(0,0,0,0)', # Transparent plot area
                        height=300,
                        margin=dict(l=40, r=40, t=40, b=40),
                        showlegend=False,
                        polar=dict(
                            radialaxis=dict(
                                visible=True, range=[0, max(radar_df['Value'].max()*1.1, 1)], # Dynamic range
                                showline=False, gridcolor=grid_color, tickfont_color=light_text_color, angle=90 # Start angle at top
                            ),
                            angularaxis=dict(
                                direction="clockwise", gridcolor=grid_color, linecolor=grid_color,
                                tickfont_color=light_text_color, tickfont_size=10
                            )
                        )
                    )
                    # Update trace appearance
                    fig_radar.update_traces(
                        line=dict(color=line_color),
                        marker=dict(color=line_color, size=6),
                        fill='toself', fillcolor=fill_color_transparent # Fill area under the line
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.warning("Could not generate radar chart (missing required data).")
        else:
            # Message when no enzyme is selected or details aren't available
            if enzyme_to_display_id:
                 # This case might occur if the selected ID was valid but the data row couldn't be fetched
                 st.warning(f"Details currently unavailable for selected enzyme ID: {enzyme_to_display_id}.")
            else:
                 # Default message when the page loads or selection is cleared
                 st.info("⬅️ Select an enzyme from the table above to view its detailed profile.")

    # --- Page 2: Property Visualizations ---
    elif page == "Property Visualizations":
        st.header("📊 Property Visualizations")
        st.markdown("""
        Explore relationships between different properties of the discovered PETase candidates.
        These visualizations help understand trade-offs and trends within the dataset.
        """)

        st.subheader("PETase Property Distribution in ESM-2 embedding space")
        st.markdown("""
        3D projection of the high-dimensional ESM-2 embedding space containing characterized marine PETases colored by selected properties.
        Select a key performance metric to visualize.
        """)
        # Selectbox for choosing which pre-generated 3D plot to show
        options_3d = list(HTML_3D_PLOT_FILES.keys())
        default_index_3d = options_3d.index("Catalytic Efficiency (M⁻¹s⁻¹)") if "Catalytic Efficiency (M⁻¹s⁻¹)" in options_3d else 0
        selected_property_3d = st.selectbox(
            "Select Property to Visualize:",
            options=options_3d,
            index=default_index_3d,
            key="3d_plot_selector"
        )

        # Load and display the selected HTML file
        selected_filename_3d = HTML_3D_PLOT_FILES.get(selected_property_3d)
        if selected_filename_3d:
            dynamic_plotly_path = IMAGE_DIR / selected_filename_3d
            if dynamic_plotly_path.is_file():
                try:
                    with open(dynamic_plotly_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    # Use Streamlit components to render the HTML
                    st.components.v1.html(html_content, height=500, scrolling=False)
                except Exception as e:
                    st.error(f"❌ Error loading 3D plot HTML file `{selected_filename_3d}`: {e}")
            else:
                st.warning(f"⚠️ 3D Plot file not found: `{selected_filename_3d}` in `{IMAGE_DIR}`")
        else:
            st.error(f"❌ Internal Error: Filename mapping missing for property: '{selected_property_3d}'")

        st.divider()
        st.subheader("Structure-Function Relationship")
        st.markdown("""
        Explore potential trade-offs, such as thermal stability vs. activity, or predicted stability vs. activity.
        * **Bubble Size:** Represents kcat (turnover rate, s⁻¹).
        * **Color:** Represents Catalytic Efficiency (M⁻¹s⁻¹), with brighter colors indicating higher efficiency.
        """)
        # Selectbox for choosing the 2D relationship plot
        plot_choice_2d = st.selectbox(
            "Select Relationship to Plot:",
            ("Thermal Stability vs. Catalytic Efficiency", "Instability Index vs. Catalytic Efficiency"),
            index=0, # Default to the first option
            key="relationship_plot_selector"
        )

        # Common plot configurations for consistency
        common_plot_cfg = {
            'height': 500,
            'color_scale': px.colors.sequential.Viridis, # Color scale for efficiency
            'opacity': 0.7,
            'size_col': 'kcat', # Use kcat for bubble size
            'color_col': 'catalytic_efficiency', # Use efficiency for color
        }
        # Base columns for hover tooltip, ensuring they exist in the dataframe
        base_hover_cols = ['short_id', 'catalytic_efficiency', 'kcat', 'km', 'tm', 'instability_index', 'temperature']
        base_hover = {k: (':.1e' if k == 'catalytic_efficiency' else ':.2f' if k in ['kcat', 'km', 'instability_index'] else ':.1f' if k in ['tm', 'temperature'] else True)
                      for k in base_hover_cols if k in df.columns}

        # Generate the selected 2D scatter plot
        fig2d = None # Initialize figure variable
        if plot_choice_2d == "Thermal Stability vs. Catalytic Efficiency":
            x_col, y_col = 'tm', 'catalytic_efficiency'
            x_label, y_label = 'Thermal Stability TM (°C)', 'Catalytic Efficiency (M⁻¹s⁻¹)'
            if x_col in df.columns and y_col in df.columns and common_plot_cfg['size_col'] in df.columns:
                hover_data = base_hover.copy()
                hover_data.pop('instability_index', None) # Remove irrelevant metric from hover
                fig2d = px.scatter(
                    df, x=x_col, y=y_col,
                    size=common_plot_cfg['size_col'], color=common_plot_cfg['color_col'],
                    color_continuous_scale=common_plot_cfg['color_scale'],
                    hover_name='short_id', hover_data=hover_data,
                    labels={x_col: x_label, y_col: y_label, common_plot_cfg['color_col']: 'Catalytic Eff.', common_plot_cfg['size_col']: 'kcat (s⁻¹)'},
                    height=common_plot_cfg['height'], opacity=common_plot_cfg['opacity'],
                    size_max=None # Let Plotly determine max bubble size based on kcat range
                    )
            else: st.warning(f"⚠️ Required columns ({x_col}, {y_col}, {common_plot_cfg['size_col']}) missing for this plot.")

        elif plot_choice_2d == "Instability Index vs. Catalytic Efficiency":
            x_col, y_col = 'instability_index', 'catalytic_efficiency'
            x_label, y_label = 'Predicted Instability Index', 'Catalytic Efficiency (M⁻¹s⁻¹)'
            if x_col in df.columns and y_col in df.columns and common_plot_cfg['size_col'] in df.columns:
                hover_data = base_hover.copy()
                hover_data.pop('tm', None) # Remove irrelevant metric from hover
                fig2d = px.scatter(
                    df, x=x_col, y=y_col,
                    size=common_plot_cfg['size_col'], color=common_plot_cfg['color_col'],
                    color_continuous_scale=common_plot_cfg['color_scale'],
                    hover_name='short_id', hover_data=hover_data,
                    labels={x_col: x_label, y_col: y_label, common_plot_cfg['color_col']: 'Catalytic Eff.', common_plot_cfg['size_col']: 'kcat (s⁻¹)'},
                    height=common_plot_cfg['height'], opacity=common_plot_cfg['opacity'],
                    size_max=None
                    )
                # Add vertical line at instability threshold
                fig2d.add_vline(x=40, line_width=2, line_dash="dash", line_color="grey",
                                annotation_text="Stable / Unstable Threshold", annotation_position="top left")
            else: st.warning(f"⚠️ Required columns ({x_col}, {y_col}, {common_plot_cfg['size_col']}) missing for this plot.")

        # Update layout and display the 2D plot if generated
        if fig2d:
            fig2d.update_layout(
                xaxis_title=x_label, yaxis_title=y_label,
                coloraxis_colorbar=dict( # Customize color bar
                    title="Catalytic Eff.", tickformat=".1e", ticks="outside",
                    bgcolor='rgba(0,0,0,0)', tickfont_color='#E0E0E0', title_font_color='#E0E0E0'
                ),
                margin=dict(l=10, r=10, t=30, b=10), # Minimal margins
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', # Transparent backgrounds
                font_color='#E0E0E0' # Light text color
            )
            st.plotly_chart(fig2d, use_container_width=True)


    # --- Page 3: Geographic Distribution ---
    elif page == "Geographic Distribution":
        st.header("🗺️ Geographic Distribution")
        st.markdown("""
        Visualize the global origins of the marine metagenomic samples from which the PETase candidates were derived.
        Understanding the source environments can offer insights into potential unique adaptations or properties of the enzymes.
        """)

        st.subheader("Global Sample Locations Map")
        st.markdown("Interactive map showing individual sample locations (dots) and highlighting countries/territories associated with samples.")
        try:
            # Generate and display the map
            geo_fig = create_geo_map_with_highlights(df)
            st.plotly_chart(geo_fig, use_container_width=True)
            st.caption("Map shows sample locations (green dots) and originating EEZ territories (yellow highlight). Hover over dots for sample details.")
        except Exception as e:
            # Catch errors during map generation
            st.error(f"❌ An error occurred while generating the geographic map: {e}")
            st.exception(e)

        st.divider()
        st.subheader("Sample Origin Overview")
        st.markdown("Breakdown of samples by their taxonomic classification (Phylum level) and geographic origin (Economic Exclusion Zone - EEZ).")

        col1, col2 = st.columns(2) # Create two columns for donut charts

        with col1:
            st.markdown("##### Taxonomic Distribution")
            if 'phylum' in df.columns:
                # Calculate counts for each phylum, filling NA with "Unknown"
                phylum_counts = df['phylum'].fillna("Unknown").value_counts()
                # Create center text for the donut chart
                phy_center_text = f"{phylum_counts.sum():,}<br>Samples"
                try:
                    # Generate and display the phylum donut chart
                    fig_phylum = create_donut_chart(phylum_counts, "Distribution by Phylum", phy_center_text)
                    st.plotly_chart(fig_phylum, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error generating Phylum distribution chart: {e}")
            else:
                # Show placeholder if phylum data is missing
                st.warning("⚠️ Phylum data column ('phylum') not available.")
                fig_phylum = create_donut_chart(pd.Series(), "Distribution by Phylum", "N/A")
                st.plotly_chart(fig_phylum, use_container_width=True)

        with col2:
            st.markdown("##### Geographic Origin (EEZ Territory)")
            # Determine which EEZ column to use, prioritizing 'eez_territory'
            country_col = None
            if 'eez_territory' in df.columns: country_col = 'eez_territory'
            elif 'eez_geoname' in df.columns: country_col = 'eez_geoname' # Fallback

            if country_col:
                # Clean and count occurrences for each territory
                country_counts = df[country_col].fillna("Unknown/Intl.") \
                                        .replace({ "Joint regime": "Joint Regime", "": "Unknown/Intl."}) \
                                        .value_counts()
                # Create center text
                cty_center_text = f"{country_counts.sum():,}<br>Samples"
                try:
                    # Generate and display the territory donut chart
                    chart_title = f"Distribution by {country_col.replace('_', ' ').title()}"
                    fig_country = create_donut_chart(country_counts, chart_title, cty_center_text)
                    st.plotly_chart(fig_country, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error generating {chart_title} chart: {e}")
            else:
                # Show placeholder if EEZ data is missing
                st.warning("⚠️ EEZ territory/geoname data not available.")
                fig_country = create_donut_chart(pd.Series(), "Distribution by Origin", "N/A")
                st.plotly_chart(fig_country, use_container_width=True)

    # --- Footer in Sidebar ---
    st.sidebar.markdown("---") # Divider before footer
    # Dynamic year for copyright
    try:
        # Use a specific timezone if relevant, otherwise rely on system time
        # Setting timezone to user's location: Spain -> 'Europe/Madrid'
        current_tz = 'Europe/Madrid'
        current_year = pd.Timestamp.now(tz=current_tz).year
    except Exception:
        # Fallback if timezone lookup fails or tz not installed
        current_year = pd.Timestamp.now().year
        # Ensure year is current or future (fallback)
        # Using the current year (2025) from the context
        current_year = max(current_year, 2025)

    st.sidebar.caption(f"© {current_year} New Atlantis Labs") # Updated copyright text

else:
    # Message displayed if the initial data loading (df = load_data()) failed
    st.error("🚨 Dashboard Cannot Be Displayed: Critical data failed to load.")
    st.warning("Please check the following:")
    st.markdown(f"""
    * Ensure required `.tsv` data files are present in the `{DATA_DIR}` directory.
    * Ensure the world shapefile (`.shp` and related files) are in `{WORLD_SHAPEFILE_PATH.parent}`.
    * Ensure any `.html` plot files are in the `{IMAGE_DIR}` directory.
    * Ensure `.pdb` files for 3D structures are in the `{PDB_DIR}` directory.
    * Run the Streamlit app (`streamlit run app.py`) from the directory containing the `app.py` file.
    """)
    # Add warning if RDKit is needed but not installed
    if not RDKIT_AVAILABLE:
        st.warning("Additionally, the **RDKit** library is required for visualizing substrate chemical structures. Please install it, for example using pip: `pip install rdkit-pypi`")