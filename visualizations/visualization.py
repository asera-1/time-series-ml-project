import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class VisualizationPipeline:
    def __init__(self, *csv_files):
        """Initialize with one or more CSV files."""
        self.csv_files = csv_files
        self.df_list = []
        
        # Read CSVs into dataframes
        for file in csv_files:
            try:
                df = pd.read_csv(file, parse_dates=['datetime'])
                self.df_list.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                self.df_list.append(None)

    def correlation_heatmap(self):
        """Generate a correlation heatmap for each dataset."""
        for idx, df in enumerate(self.df_list, 1): 
            if df is not None:
                df_numeric = df.select_dtypes(include=['number'])
                corr_matrix = df_numeric.corr()

                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cbar=False, cmap='Blues', fmt='.1f')
                plt.title(f'Correlation Heatmap for Dataset {idx}')
                plt.show()

    def plot_avg_sales_per_day_of_week(self):
        """Plot average sales per day of the week for one or two stores."""
        def process_store(df):
            df['weekday'] = df['datetime'].dt.weekday
            df['total_sales'] = df.iloc[:, 1:].sum(axis=1)
            return df.groupby('weekday')['total_sales'].mean()

        avg_sales1 = process_store(self.df_list[0])
        avg_sales2 = process_store(self.df_list[1]) if self.df_list[1] is not None else None

        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        avg_sales1.index = [weekday_names[i] for i in avg_sales1.index]
        if avg_sales2 is not None:
            avg_sales2.index = [weekday_names[i] for i in avg_sales2.index]

        plt.figure(figsize=(6, 6))
        if avg_sales2 is not None:
            plt.bar(avg_sales2.index, avg_sales2, width=0.4, label="Store 2", color='lightgreen', edgecolor='black')

        plt.bar(avg_sales1.index, avg_sales1, width=0.4, label="Store 1", color='skyblue', edgecolor='black')

        plt.title('Average Sales per Day of the Week')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    def plot_total_items_per_day(self):
        """Plot total items sold per day for both stores."""
        daily_totals1 = self.prepare_daily_totals(self.df_list[0])
        daily_totals2 = self.prepare_daily_totals(self.df_list[1])
        
        # Plot daily totals
        self.plot_daily_totals(daily_totals1, daily_totals2)

    def prepare_daily_totals(self, df):
        """Prepare daily totals for a given dataframe."""
        if df is None:
            return None
        
        df['date'] = df['datetime'].dt.date
        return df.groupby('date').sum(numeric_only=True)

    def plot_daily_totals(self, daily_totals1, daily_totals2):
        """Plot total items sold per day for both stores."""
        plt.figure(figsize=(12, 5))
        
        if daily_totals1 is not None:
            plt.plot(daily_totals1.index, daily_totals1.sum(axis=1), linestyle='-', color='b', label='Store 1')
        if daily_totals2 is not None:
            plt.plot(daily_totals2.index, daily_totals2.sum(axis=1), linestyle='-', color='orange', label='Store 2')

        plt.title("Total Items Sold Per Day")
        plt.xlabel("Date")
        plt.ylabel("Total Items Sold")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_sales_per_category(self):
        """Plot daily aggregated sales for each category in all datasets."""
        sales_columns = [
            'MAIN', 'SIDE', 'SOUP', 'DESSERT', 'SALAD', 'BOTTLE', 'BAKED_GOOD', 'CONDIMENT', 'OTHER'
        ]
        
        for idx, df in enumerate(self.df_list, 1):  # Loop over each dataset
            if df is not None:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['date'] = df['datetime'].dt.date
                df['weekday'] = df['datetime'].dt.weekday
                df = df[df['weekday'] < 5]  # Filter out weekends (Saturday, Sunday)

                # Aggregate sales per category by date
                df_daily = df.groupby('date')[sales_columns].sum()

                # Plot sales for each category
                num_categories = len(sales_columns)
                fig, axes = plt.subplots(num_categories, 1, figsize=(12, 2 * num_categories))
                
                for i, category in enumerate(sales_columns):
                    axes[i].plot(df_daily.index, df_daily[category], label=category)
                    if i == 0: 
                        axes[i].set_title(f'Sales for {category} Over Time - Dataset {idx}')
                    else: 
                        axes[i].set_title(f'Sales for {category}')
                    
                    # Only show x-axis label for the last category
                    if i == len(sales_columns) - 1:
                        axes[i].set_xlabel('Date')
                    else:
                        axes[i].set_xticklabels([])

                    axes[i].set_ylabel('Sales')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(True)
                    axes[i].legend()

                plt.subplots_adjust(hspace=0.2)
                plt.tight_layout()
                plt.show()

    def process_and_plot_sales(self):
        """Process and plot sales per category for each dataset."""
        for idx, df in enumerate(self.df_list, 1):  # Loop over each dataset
            if df is not None:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['date'] = df['datetime'].dt.date
                df['weekday'] = df['datetime'].dt.weekday
                df = df[df['weekday'] < 5]  # Keep only weekdays (0-4)

        self.plot_sales_per_category()

    def plot_total_sales_with_smoothing(self, smoothing_window=7):
        """Plot total sales with smoothing (Exponential Moving Average) for both stores."""
        def process_data(df):
            """Process data by removing weekends and calculating total sales."""
            if df is not None:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['date'] = df['datetime'].dt.date
                df['weekday'] = df['datetime'].dt.weekday
                df = df[df['weekday'] < 5]  # Exclude Saturday (5) and Sunday (6)
                df.loc[:, 'total_sales'] = df[['MAIN', 'SIDE', 'SOUP', 'DESSERT', 'SALAD', 'BOTTLE', 'BAKED_GOOD', 'CONDIMENT', 'OTHER']].sum(axis=1)
                daily_sales = df.groupby('date')['total_sales'].sum()
                return daily_sales
            return None

        # Process both stores
        daily_sales_store1 = process_data(self.df_list[0])
        daily_sales_store2 = process_data(self.df_list[1]) if len(self.df_list) > 1 else None

        # Combine both stores into one DataFrame
        combined_sales = pd.concat([daily_sales_store1, daily_sales_store2], axis=1)
        combined_sales.columns = ['Store 1', 'Store 2']

        # Apply Exponential Moving Average (EMA) for smoothing
        smoothed_sales_store1 = combined_sales['Store 1'].ewm(span=smoothing_window, adjust=False).mean()
        smoothed_sales_store2 = combined_sales['Store 2'].ewm(span=smoothing_window, adjust=False).mean()

        # Plot the data and smoothed fit
        plt.figure(figsize=(12, 5))

        if smoothed_sales_store1 is not None:
            plt.plot(smoothed_sales_store1.index, smoothed_sales_store1.values, label=f'Store 1 Smoothed Fit (EMA, window={smoothing_window})', linestyle='-', color='red', linewidth=2)

        if smoothed_sales_store2 is not None:
            plt.plot(smoothed_sales_store2.index, smoothed_sales_store2.values, label=f'Store 2 Smoothed Fit (EMA, window={smoothing_window})', linestyle='-', color='orange', linewidth=2)

        plt.title('Total Items Sold per Day with Smoothed Fit (Excluding Weekends)')
        plt.xlabel('Date')
        plt.ylabel('Total Items Sold')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def process_sales_data(self, df, valid_intervals, weekday_names):
        """Process sales data by weekday and time interval."""
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['weekday'] = df['datetime'].dt.weekday
        df['time_interval'] = df['datetime'].dt.strftime('%H:%M')
        df['total_sales'] = df[['MAIN', 'SIDE', 'SOUP', 'DESSERT', 'SALAD', 'BOTTLE', 'BAKED_GOOD', 'CONDIMENT', 'OTHER']].sum(axis=1)
        df = df[df['time_interval'].isin(valid_intervals)]
        all_combinations = pd.MultiIndex.from_product([range(7), valid_intervals], names=['weekday', 'time_interval'])
        avg_sales = df.groupby(['weekday', 'time_interval'])['total_sales'].mean().reindex(all_combinations, fill_value=0)
        avg_sales.index = [(weekday_names[day], interval) for day, interval in avg_sales.index]
        return avg_sales

    def plot_avg_sales_per_day_and_interval_bar(self):
        """Plot average sales per day and time interval for each store."""
        valid_intervals = ["11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45", "14:00"]
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Process both stores
        avg_sales_store1 = self.process_sales_data(self.df_list[0], valid_intervals, weekday_names) if self.df_list[0] is not None else None
        avg_sales_store2 = self.process_sales_data(self.df_list[1], valid_intervals, weekday_names) if len(self.df_list) > 1 and self.df_list[1] is not None else None

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if avg_sales_store1 is not None:
            avg_sales_store1.plot(kind='bar', width=0.4, position=1, ax=ax, color='skyblue', edgecolor='black', label='Store 1')
        if avg_sales_store2 is not None:
            avg_sales_store2.plot(kind='bar', width=0.4, position=0, ax=ax, color='lightgreen', edgecolor='black', label='Store 2')

        ax.set_title('Average Sales per Day of the Week and Time Interval for Store 1 and Store 2')
        ax.set_xlabel('Day of the Week and Time Interval')
        ax.set_ylabel('Average Total Items Sold')

        days = ["mo", "tu", "we", "th", "fr", "sa", "su"]
        # Create custom tick labels for all combinations
        tick_labels = [f'{day} {interval}' for day in days for interval in valid_intervals]
        ax.set_xticks(range(0, 77))  # Set tick positions for each combination
        ax.set_xticklabels(tick_labels, rotation=90)

        ax.legend()
        plt.tight_layout()
        plt.show()

    def run_pipeline(self, functions):
        """Run selected visualization functions."""
        for func_name in functions:
            if hasattr(self, func_name):
                print(f"Running: {func_name}...")
                getattr(self, func_name)()
            else:
                print(f"Function {func_name} not found.")


# Example Usage:
pipeline = VisualizationPipeline("data/cleaned_store1.csv", "data/cleaned_store2.csv")
pipeline.run_pipeline(["plot_avg_sales_per_day_of_week", "plot_avg_sales_per_day_and_interval_bar", "process_and_plot_sales", "plot_total_items_per_day", "plot_total_sales_with_smoothing", "correlation_heatmap", "plot_sales_per_category"])