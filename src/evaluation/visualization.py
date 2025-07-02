import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_distributions(df, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    sns.histplot(df['demand'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Demand')
    sns.histplot(df['RRP'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of RRP')
    sns.histplot(df['max_temperature'], kde=True, ax=axes[0, 2])
    axes[0, 2].set_title('Distribution of Max Temperature')
    sns.histplot(df['solar_exposure'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Solar Exposure')
    sns.histplot(df['rainfall'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Rainfall')
    sns.histplot(df['min_temperature'], kde=True, ax=axes[1, 2])
    axes[1, 2].set_title('Distribution of Min Temperature')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_correlation_heatmap(df, save_path=None):
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_actual_vs_pred(y_true, y_pred, model_name, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label='Predicted')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'g--', label='Ideal')
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.title(f'Actual vs Predicted Demand ({model_name})')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_year_vs_demand(df, y_pred, model_name, save_path=None):
    df = df.copy()
    df['predicted_demand'] = y_pred
    yearly_actual = df.groupby('year')['demand'].mean()
    yearly_pred = df.groupby('year')['predicted_demand'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_actual.index, yearly_actual.values, marker='o', label='Actual Demand')
    plt.plot(yearly_pred.index, yearly_pred.values, marker='o', linestyle='--', label='Predicted Demand')
    plt.title(f'Year vs. Demand (Actual vs. Predicted) - {model_name}')
    plt.xlabel('Year')
    plt.ylabel('Demand')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_demand_timeseries(df, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df['date']), df['demand'])
    plt.title('Time Series of  Demand')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_demand_by_dayofweek(df, save_path=None):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='day_of_week', y='demand', data=df)
    plt.title('Demand by Day of Week')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_demand_vs_maxtemp(df, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['max_temperature'], df['demand'])
    plt.title('Demand vs Max Temperature')
    plt.xlabel('max_temperature')
    plt.ylabel('demand')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_demand_school_vs_nonschool(df, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='school_day', y='demand', data=df)
    plt.title('Demand on School Days vs Non-School Days')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_monthly_avg_demand(df, save_path=None):
    monthly = df.copy()
    monthly['date'] = pd.to_datetime(monthly['date'])
    monthly = monthly.groupby(monthly['date'].dt.to_period('M')).demand.mean().reset_index()
    monthly['date'] = monthly['date'].astype(str)
    plt.figure(figsize=(14, 7))
    plt.plot(monthly['date'], monthly['demand'])
    plt.title('Monthly Average Demand')
    plt.xlabel('Date')
    plt.ylabel('Average Demand')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_demand_vs_rrp(df, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['demand'], df['RRP'], alpha=0.5)
    plt.title('Demand vs. RRP')
    plt.xlabel('Demand')
    plt.ylabel('RRP')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_avg_demand_by_month(df, save_path=None):
    plt.figure(figsize=(12, 6))
    df['month'] = pd.to_datetime(df['date']).dt.month
    avg_by_month = df.groupby('month').demand.mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(months, avg_by_month)
    plt.title('Average Demand by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Demand')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_demand_vs_rrp_joint(df, save_path=None):
    g = sns.jointplot(x='RRP', y='demand', data=df, kind='hex')
    g.set_axis_labels('RRP', 'demand')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_demand_vs_solar(df, save_path=None):
    plt.figure(figsize=(12, 8))
    plt.scatter(df['solar_exposure'], df['demand'], alpha=0.5)
    plt.title('Demand vs Solar Exposure')
    plt.xlabel('Solar Exposure')
    plt.ylabel('Demand')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_demand_distribution_split(train, val, test, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].hist(train['demand'], bins=30)
    axes[0].set_title('Training Data Distribution')
    axes[0].set_xlabel('Demand')
    axes[0].set_ylabel('Frequency')
    axes[1].hist(val['demand'], bins=30)
    axes[1].set_title('Validation Data Distribution')
    axes[1].set_xlabel('Demand')
    axes[2].hist(test['demand'], bins=30)
    axes[2].set_title('Test Data Distribution')
    axes[2].set_xlabel('Demand')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_demand_distribution_box_split(train, val, test, save_path=None):
    plt.figure(figsize=(12, 7))
    data = [train['demand'], val['demand'], test['demand']]
    plt.boxplot(data, labels=['Training', 'Validation', 'Test'])
    plt.title('Demand Distribution Across Sets')
    plt.ylabel('Demand')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close() 