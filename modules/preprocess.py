import pandas as pd

# ----------------------------
# Load the CSV and preprocess dates
# ----------------------------
def load_csv(csv_path):
    """
    Loads the Global Mobility CSV dataset and adds a 'year' column.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'year' extracted from the 'date' column.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime
    df['year'] = df['date'].dt.year  # Extract year for filtering
    return df

# ----------------------------
# Categorize mobility values into states
# ----------------------------
def categorize_states(series):
    """
    Categorizes numeric mobility data into qualitative labels: Low, Moderate, High.

    Args:
        series (pd.Series): A numeric Pandas Series (e.g., % change in mobility)

    Returns:
        pd.Series: A categorical Series with values: 'Low', 'Moderate', 'High'
    """
    bins = [-float('inf'), -20, 5, float('inf')]  # Mobility thresholds
    labels = ['Low', 'Moderate', 'High']          # Category labels
    return pd.cut(series, bins=bins, labels=labels)

# ----------------------------
# Filter by country/year and convert to mobility states
# ----------------------------
def get_mobility_states(df, country, year, category):
    """
    Extracts and categorizes mobility data into states for a specific country and year.

    Args:
        df (pd.DataFrame): The loaded mobility DataFrame.
        country (str): Country name to filter.
        year (int): Year to filter (e.g., 2021).
        category (str): Column name for mobility type (e.g., 'retail_and_recreation_percent_change_from_baseline').

    Returns:
        list: A list of states (Low/Moderate/High) representing daily mobility behavior.
    """
    filtered = df[(df['country_region'] == country) & (df['year'] == year)]
    filtered = filtered.dropna(subset=[category])  # Ensure no missing data
    states = categorize_states(filtered[category]).dropna().tolist()
    return states

# ----------------------------
# Load + extract states in one call (shortcut)
# ----------------------------
def get_state_sequence(csv_path, column, country='Pakistan', year=2021):
    """
    Loads data and returns categorized mobility states in one step.

    Args:
        csv_path (str): CSV file path
        column (str): Mobility category column name
        country (str): Default is 'Pakistan'
        year (int): Default is 2021

    Returns:
        list: State sequence (Low, Moderate, High)
    """
    df = load_csv(csv_path)
    return get_mobility_states(df, country, year, column)
