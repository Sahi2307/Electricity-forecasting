import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.data.data_loader import load_raw_data
from src.data.preprocessor import preprocess_data
from src.evaluation.visualization import (
    plot_distributions, plot_correlation_heatmap, plot_demand_timeseries, plot_demand_by_dayofweek,
    plot_demand_vs_maxtemp, plot_demand_school_vs_nonschool, plot_monthly_avg_demand, plot_demand_vs_rrp,
    plot_avg_demand_by_month, plot_demand_vs_rrp_joint, plot_demand_vs_solar,
    plot_demand_distribution_split, plot_demand_distribution_box_split
)
from sklearn.model_selection import train_test_split

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    df = load_raw_data(config)

    # Preprocess data
    df_processed, scaler, le = preprocess_data(df, config)

    # EDA visualizations
    eda_dir = config['paths']['figures'] + 'eda/'
    os.makedirs(eda_dir, exist_ok=True)
    plot_distributions(df_processed, save_path=os.path.join(eda_dir, 'distributions.png'))
    plot_correlation_heatmap(df_processed, save_path=os.path.join(eda_dir, 'correlation_heatmap.png'))
    plot_demand_timeseries(df_processed, save_path=os.path.join(eda_dir, 'demand_timeseries.png'))
    plot_demand_by_dayofweek(df_processed, save_path=os.path.join(eda_dir, 'demand_by_dayofweek.png'))
    plot_demand_vs_maxtemp(df_processed, save_path=os.path.join(eda_dir, 'demand_vs_maxtemp.png'))
    plot_demand_school_vs_nonschool(df_processed, save_path=os.path.join(eda_dir, 'demand_school_vs_nonschool.png'))
    plot_monthly_avg_demand(df_processed, save_path=os.path.join(eda_dir, 'monthly_avg_demand.png'))
    plot_demand_vs_rrp(df_processed, save_path=os.path.join(eda_dir, 'demand_vs_rrp.png'))
    plot_avg_demand_by_month(df_processed, save_path=os.path.join(eda_dir, 'avg_demand_by_month.png'))
    plot_demand_vs_rrp_joint(df_processed, save_path=os.path.join(eda_dir, 'demand_vs_rrp_joint.png'))
    plot_demand_vs_solar(df_processed, save_path=os.path.join(eda_dir, 'demand_vs_solar.png'))

    # Train/Val/Test split for split-based plots
    train, temp = train_test_split(df_processed, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    plot_demand_distribution_split(train, val, test, save_path=os.path.join(eda_dir, 'demand_distribution_split.png'))
    plot_demand_distribution_box_split(train, val, test, save_path=os.path.join(eda_dir, 'demand_distribution_box_split.png'))

if __name__ == '__main__':
    main() 