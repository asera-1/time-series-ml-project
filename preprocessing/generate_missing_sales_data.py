import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def first_time_sold(df, column_name):
    """Finds the index of the first nonzero value in the specified column."""
    nonzero_indices = df[df[column_name] != 0].index
    return nonzero_indices[0] if not nonzero_indices.empty else None


def trim_after_first_time_sold(df, column_name):
    """Trims the data from the first nonzero occurrence onward and returns the trimmed dataframe."""
    first_index = first_time_sold(df, column_name)

    if first_index is not None:
        trimmed_df = df.iloc[first_index:]  # Keep the data from the first non-zero row
        return trimmed_df
    return "No nonzero values found in the column."

def adjust_other_column(df):
    """Substract the imputed values of condiment and baked goods from category: 'Other'"""
    if 'OTHER' in df.columns and 'BAKED_GOOD' in df.columns and 'CONDIMENT' in df.columns:
        df['OTHER'] = df['OTHER'] - df['BAKED_GOOD'] - df['CONDIMENT']
        df['OTHER'] = df['OTHER'].apply(lambda x: max(x, 0))  # Ensure no negative values
    return df


def generate_missing_data(df, category_column, store):
    """Generates realistic sales data for days on which sales for a specific category haven't been monitored."""

    df['datetime'] = pd.to_datetime(df['datetime'])

    # Identify category columns
    category_columns = df.columns[df.columns.str.contains(category_column)]
    for column_name in category_columns:
        zero_sales_mask = df[column_name] == 0

        if zero_sales_mask.any():
            df_numeric = df.drop(columns=['datetime'])
            corr_matrix = df_numeric.corr()
            # plot correlation matrix
            corr_matrix.to_csv(f"corr_matrix_{store}_{column_name}.csv")

            relevant_columns = corr_matrix[column_name].drop(column_name).sort_values(ascending=False).index[:6]
            print(f"Relevant columns for {column_name}: {relevant_columns}")

            valid_data = df[~zero_sales_mask]
            X_train = valid_data[relevant_columns]
            y_train = valid_data[column_name]

            # Train linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            X_missing = df[zero_sales_mask][relevant_columns]
            predicted_sales = model.predict(X_missing)
            predicted_sales_int = np.round(predicted_sales).astype(int)

            # Impute the predicted sales values as integers
            df.loc[zero_sales_mask, column_name] = predicted_sales_int
            print(f"Imputed sales data for {column_name}.")

    return df


def adjust_zero_sales(df, category_column):
    """Adjusts the data where total_sales is zero, and sets corresponding product columns to zero.
    Also ensures that there are no negative values, replacing them with zero."""

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.loc[df['TOTAL_SALES'] == 0, category_column] = 0

    # Replace negative values
    df.loc[:, df.columns.difference(['datetime', 'TOTAL_SALES'])] = df.loc[:, df.columns.difference(['datetime', 'TOTAL_SALES'])].map(lambda x: max(x, 0))

    return df


def merge_and_save(df1, df2, final_file):
    """
    Replaces the BAKED_GOOD column in df1 with the BAKED_GOOD column from df2 and saves the updated dataset.
    
    Parameters:
    - df1: The first DataFrame (usually the condiment data).
    - df2: The second DataFrame (usually the baked good data).
    - final_file: The path where the merged and modified dataset will be saved.
    """
    # Replace the BAKED_GOOD column in df1 with the BAKED_GOOD column from df2
    df1['BAKED_GOOD'] = df2['BAKED_GOOD']

    df1.to_csv(final_file, index=False)
    print(f"Updated data saved to {final_file}.")

def adjust_total_sales(df):
    columns = df.columns.difference(['datetime', 'TOTAL_SALES'])  # All columns except 'time'
    df['TOTAL_SALES'] = df[columns].sum(axis=1)
    return df

def process_sales_data(data_dir, cleaned_data_file, category_columns, store):
    """
    Processes sales data by performing trimming, missing data generation, and zero sales adjustment
    for both CONDIMENT and BAKED_GOOD categories for a single cleaned data file.
    
    Parameters:
    - cleaned_data_file: The cleaned data file name (e.g., "cleaned_data1.csv" or "cleaned_data2.csv")
    - category_columns: List of category column names (e.g., ["CONDIMENT", "BAKED_GOOD"])
    - store: The store number (1 or 2) for naming the final output files.
    """
    processed_data = {}

    df = pd.read_csv(f"{data_dir}/{cleaned_data_file}")
    
    # Trim data after the first time sold for each category
    for category in category_columns:
        trimmed_data = trim_after_first_time_sold(df, category)  # Pass the dataframe directly
        processed_data[(cleaned_data_file, category, 'trimmed')] = trimmed_data
    
    # Generate missing data for each cleaned data file and category
    for category in category_columns:
        for idx in [1, 2]:  # Assuming index values 1 and 2 for generating different versions
            generated_missing_data = generate_missing_data(df, category, idx)
            processed_data[(cleaned_data_file, category, f'missing_data_{idx}')] = generated_missing_data
    
    # Adjust zero sales for generated sales data
    for category in category_columns:
        for idx in [1, 2]:
            generated_sales_data = processed_data[(cleaned_data_file, category, f'missing_data_{idx}')]
            adjusted_sales_data = adjust_zero_sales(generated_sales_data, category)
            processed_data[(cleaned_data_file, category, f'adjusted_sales_{idx}')] = adjusted_sales_data
    
    # Merge the processed data for both CONDIMENT and BAKED_GOOD for a store
    # Merge into a single DataFrame
    condiment_adjusted = processed_data[(cleaned_data_file, "CONDIMENT", "adjusted_sales_1")]
    bakedgood_adjusted = processed_data[(cleaned_data_file, "BAKED_GOOD", "adjusted_sales_1")]

    final_merged_data = condiment_adjusted.copy() 
    final_merged_data['BAKED_GOOD'] = bakedgood_adjusted['BAKED_GOOD']

    final_merged_data = adjust_other_column(final_merged_data) 
    final_merged_data = adjust_total_sales(final_merged_data)
    
    final_file = f"{data_dir}/cleaned_adjusted_store{store}.csv"
    final_merged_data.to_csv(final_file, index=False)
    print(f"Final merged data saved to {final_file}.")


# Example usage
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
cleaned_data_file = "cleaned_store1.csv"  # or "cleaned_data2.csv"
category_columns = ["CONDIMENT", "BAKED_GOOD"]

# process_sales_data(data_dir, cleaned_data_file, category_columns, store=1)
process_sales_data(data_dir, cleaned_data_file, category_columns, store=1)

