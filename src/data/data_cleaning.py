import pandas as pd


# Reusable function
def load_country_data(country_name):

    # Load dataset
    df = pd.read_csv("data/raw/who_covid_data.csv")

    # Keep important columns
    df = df[
        [
            "Date_reported",
            "Country",
            "New_cases",
            "Cumulative_cases",
            "New_deaths",
            "Cumulative_deaths"
        ]
    ]

    # Convert date column to datetime
    df["Date_reported"] = pd.to_datetime(df["Date_reported"])

    # Fill missing values
    df["New_cases"] = df["New_cases"].fillna(0)
    df["New_deaths"] = df["New_deaths"].fillna(0)

    # Remove remaining missing rows
    df = df.dropna()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Filter specific country
    country_df = df[df["Country"] == country_name].copy()

    # Sort chronologically
    country_df = country_df.sort_values("Date_reported")

    # Reset index
    country_df = country_df.reset_index(drop=True)

    # Create complete date range
    full_dates = pd.date_range(
        start=country_df["Date_reported"].min(),
        end=country_df["Date_reported"].max()
    )

    # Set date column as index
    country_df = country_df.set_index("Date_reported")

    # Reindex with full dates
    country_df = country_df.reindex(full_dates)

    # Rename index
    country_df.index.name = "Date_reported"

    # Fill missing daily values
    country_df["New_cases"] = country_df["New_cases"].fillna(0)
    country_df["New_deaths"] = country_df["New_deaths"].fillna(0)

    # Forward-fill cumulative values
    country_df["Cumulative_cases"] = (
        country_df["Cumulative_cases"].ffill()
    )

    country_df["Cumulative_deaths"] = (
        country_df["Cumulative_deaths"].ffill()
    )

    # Create 7-day rolling average
    country_df["Cases_7day_avg"] = (
        country_df["New_cases"]
        .rolling(window=7)
        .mean()
    )

    # Create growth rate feature
    country_df["Growth_Rate"] = (
        country_df["New_cases"]
        .pct_change()
        * 100
    )

    # Replace negative values with 0
    country_df["New_cases"] = (
        country_df["New_cases"]
        .clip(lower=0)
    )

    # Prepare Prophet dataframe
    prophet_df = country_df.reset_index()[
        ["Date_reported", "New_cases"]
    ]

    # Rename columns for Prophet
    prophet_df.columns = ["ds", "y"]

    return prophet_df


# Load India dataset
prophet_df = load_country_data("India")

# Preview Prophet dataset
print(prophet_df.head())

# Dataset info
print(prophet_df.info())

# Save cleaned dataset
prophet_df.to_csv(
    "data/processed/india_covid_clean.csv",
    index=False
)

print("Processed dataset saved successfully!")