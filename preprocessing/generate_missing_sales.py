import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def first_time_sold(df, column_name):
    """Finds the index of the first nonzero value in the specified column."""
    nonzero_indices = df[df[column_name] != 0].index
    if nonzero_indices.empty:
        return None
    else:
        return df.loc[nonzero_indices[0], 'datetime']

def generate_missing_data(df, column_name, category_columns):
    """Generates realistic sales data for days on which sales for a specific category haven't been monitored."""
    
    cutoff_date = first_time_sold(df, column_name)
    print(f"First time sold: {cutoff_date}")
    df['datetime'] = pd.to_datetime(df['datetime'])

    df_cutoff = df[df['datetime'] < cutoff_date]
    df_after_cutoff = df[df['datetime'] >= cutoff_date]

    df_after_cutoff.loc[:,"OTHER"] += df_after_cutoff[category_columns].sum(axis=1)

    print(f"Post cutoff, pre imputation mean:{df_after_cutoff[column_name].mean()} and std:{df_after_cutoff[column_name].std()}")
    # Identify category columns
    df_numeric = df_after_cutoff.drop(columns=['datetime'])
    corr_matrix = df_numeric.corr()
    print(f"Sorted correlation values for {column_name}: {corr_matrix[column_name].drop(category_columns).sort_values(ascending=False)}") 
    relevant_columns = corr_matrix[column_name].drop(category_columns).sort_values(ascending=False).index[:7]
    print(f"Relevant columns for {column_name}: {relevant_columns}")

    X_train = df_after_cutoff[relevant_columns]
    y_train = df_after_cutoff[column_name]

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    X_missing = df_cutoff[relevant_columns]
    predicted_sales = model.predict(X_missing)
    print(f"Post cutoff, post_umputation mean: {np.mean(predicted_sales)} and std: {np.std(predicted_sales)}")
    df_cutoff.loc[:, column_name] = np.round(predicted_sales).astype(int)
    df_cutoff.loc[:, column_name] = df_cutoff[column_name].apply(lambda x: max(x, 0))
    
    df = pd.concat([df_cutoff, df_after_cutoff])
    print(f"Post imputation mean:{df[column_name].mean()} and std:{df[column_name].std()}") 
    print(f"Imputed sales data for {column_name}.")
    return df

def adjust_zero_sales(df, category_column):
    """Adjusts the data where total_sales is zero, and sets corresponding product columns to zero.
    Also ensures that there are no negative values, replacing them with zero."""

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.loc[df['TOTAL_SALES'] == 0, category_column] = 0
    return df

def adjust_other_column(df, category_column):
    """Substract the imputed values of condiment and baked goods from category: 'Other'"""
    if 'OTHER' in df.columns:
        df['OTHER'] = df['OTHER'] - df[category_column]
        df['OTHER'] = df['OTHER'].apply(lambda x: max(x, 0))  # Ensure no negative values
    return df

def adjust_total_sales(df):
    columns = df.columns.difference(['datetime', 'TOTAL_SALES'])  # All columns except 'time'
    df['TOTAL_SALES'] = df[columns].sum(axis=1)
    return df

def plot_processed_sales(df, processed_df, category_columns, store):
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    processed_df["date"] = pd.to_datetime(processed_df["datetime"]).dt.date
    fig, ax = plt.subplots(len(category_columns), 2, figsize=(15, 2 * len(category_columns)))
    for i, category in enumerate(category_columns):
        ax[i, 0].plot(df['date'], df[category])
        ax[i, 0].set_title(f'{category} before processing')
        ax[i, 1].plot(processed_df['date'], processed_df[category])
        ax[i, 1].set_title(f'{category} after processing')
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/processed_sales{store}.png")
    plt.close(fig)

def process_sales_data(data_dir, cleaned_data_file, category_columns, store):
    """
    Processes sales data by performing trimming, missing data generation, and zero sales adjustment
    for both CONDIMENT and BAKED_GOOD categories for a single cleaned data file.
    
    Parameters:
    - cleaned_data_file: The cleaned data file name (e.g., "cleaned_data1.csv" or "cleaned_data2.csv")
    - category_columns: List of category column names (e.g., ["CONDIMENT", "BAKED_GOOD"])
    - store: The store number (1 or 2) for naming the final output files.
    """

    df = pd.read_csv(f"{data_dir}/{cleaned_data_file}")
    
    processed_df = df.copy()

    for category in category_columns:
        category_df = df.copy()
        category_df = generate_missing_data(category_df, category, category_columns)
        adjust_zero_sales(category_df, category)
        processed_df[category] = category_df[category]

    for category in category_columns:
        processed_df = adjust_other_column(processed_df, category)
    processed_df = adjust_total_sales(processed_df)

    final_file = f"{data_dir}/cleaned_adjusted_store{store}.csv"
    processed_df.to_csv(final_file, index=False)
    print(f"Final data saved to {final_file}.")
    plot_processed_sales(df, processed_df, category_columns + ['OTHER'], store)
    return processed_df

# Example usage
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
for store in [1, 2]:
    cleaned_data_file = f"cleaned_store{store}.csv"  # or "cleaned_data2.csv"
    category_columns = ["CONDIMENT", "BAKED_GOOD"]
    # process_sales_data(data_dir, cleaned_data_file, category_columns, store=1
    processed_df = process_sales_data(data_dir, cleaned_data_file, category_columns, store=store)

