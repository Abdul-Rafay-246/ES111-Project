# =============================================================================
# Cricket Match Analysis Script
# This script performs comprehensive statistical analysis of cricket match data
# including format statistics, match outcomes, team performance, and home advantage
# =============================================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2
import re
import os

# Set matplotlib backend for proper visualization
plt.switch_backend('TkAgg')

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the cricket dataset.
    pandas
numpy
matplotlib
scipy
    Parameters:
    - file_path: Path to the CSV file containing cricket match data
    
    Returns:
    - df: Original dataframe with all match data
    - df_runs: Processed dataframe focusing on runs-based matches
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Create runs dataframe by extracting numeric margin values
    df_runs = df.copy()
    df_runs['Margin'] = df_runs['Margin'].str.extract(r'(\d+)\s*(?:run|runs)').astype(float)
    df_runs.dropna(subset=['Margin'], inplace=True)
    
    return df, df_runs

def parse_margin(margin):
    """
    Parses match margin into numeric value and outcome type.
    
    Parameters:
    - margin: String containing match margin information
    
    Returns:
    - Tuple of (margin_value, outcome_type)
    """
    margin = str(margin).lower()
    runs = re.findall(r'(\d+)\s*run', margin)
    wickets = re.findall(r'(\d+)\s*wicket', margin)
    if runs: return float(runs[0]), 'Won by Runs'
    if wickets: return float(wickets[0]), 'Won by Wickets'
    if 'drawn' in margin: return np.nan, 'Drawn'
    if 'no result' in margin: return np.nan, 'No Result'
    if 'tied' in margin: return np.nan, 'Tied'
    return np.nan, 'Other'

def calculate_format_statistics(df_runs):
    """
    Calculates and returns format-wise statistics including means and variances.
    
    Parameters:
    - df_runs: Dataframe containing runs-based match data
    
    Returns:
    - DataFrame with comparative statistics for each format
    """
    results = []
    for format_name, group in df_runs.groupby('Format'):
        data = group['Margin']
        direct_mean, direct_var = data.mean(), data.var(ddof=1)
        
        # Calculate frequency distribution statistics
        counts, bins = np.histogram(data, bins=20)
        bin_midpoints = (bins[:-1] + bins[1:]) / 2
        freq_mean = np.sum(counts * bin_midpoints) / counts.sum()
        freq_var = np.sum(counts * (bin_midpoints - freq_mean)**2) / (counts.sum() - 1)
        
        results.append({
            'Format': format_name,
            'Direct Mean': direct_mean,
            'Frequency Mean': freq_mean,
            'Mean Difference': abs(direct_mean - freq_mean),
            'Direct Variance': direct_var,
            'Frequency Variance': freq_var,
            'Variance Difference': abs(direct_var - freq_var)
        })
    return pd.DataFrame(results)

def plot_match_outcomes(df):
    """
    Creates a pie chart visualization of match outcomes.
    
    Parameters:
    - df: Dataframe containing match data with outcome types
    """
    plt.figure(figsize=(10, 6))
    outcome_counts = df['Outcome_Type'].value_counts()
    if not outcome_counts.empty:
        plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%',
                startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0'])
        plt.title('Match Outcome Distribution')
        plt.show()

def plot_margin_histograms(df):
    """
    Creates histograms for runs and wickets margins.
    
    Parameters:
    - df: Dataframe containing match data with margin values
    """
    plt.figure(figsize=(12, 5))
    
    # Plot runs margin histogram
    runs_data = df[df['Outcome_Type'] == 'Won by Runs']['Margin_Value'].dropna()
    if not runs_data.empty:
        plt.subplot(1, 2, 1)
        counts, bins, _ = plt.hist(runs_data, bins=20, edgecolor='black', color='blue')
        plt.title('Winning Margins by Runs')
        plt.xlabel('Runs')
        plt.ylabel('Frequency')
        
        # Calculate and display mean values
        bin_midpoints = (bins[:-1] + bins[1:])/2
        freq_mean = np.sum(counts * bin_midpoints)/counts.sum()
        print(f"\nRuns Margin Mean (Frequency vs Direct): {freq_mean:.2f} vs {runs_data.mean():.2f}")
    
    # Plot wickets margin histogram
    wickets_data = df[df['Outcome_Type'] == 'Won by Wickets']['Margin_Value'].dropna()
    if not wickets_data.empty:
        plt.subplot(1, 2, 2)
        plt.hist(wickets_data, bins=10, edgecolor='black', color='green')
        plt.title('Winning Margins by Wickets')
        plt.xlabel('Wickets Remaining')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def analyze_team_performance(df):
    """
    Analyzes and visualizes team performance by victory type.
    
    Parameters:
    - df: Dataframe containing match data with team information
    """
    # Define list of teams to analyze
    teams = ['India', 'Pakistan', 'Australia', 'England', 'New Zealand', 
             'South Africa', 'Sri Lanka', 'West Indies', 'Bangladesh', 
             'Afghanistan', 'Zimbabwe', 'Ireland', 'ICC World XI']
    
    # Calculate win statistics for each team
    win_data = []
    for team in teams:
        team_wins = df[df['Winner'] == team]
        win_data.append({
            'Team': team,
            'Won by Runs': team_wins[team_wins['Outcome_Type'] == 'Won by Runs'].shape[0],
            'Won by Wickets': team_wins[team_wins['Outcome_Type'] == 'Won by Wickets'].shape[0]
        })
    
    # Create and display bar chart
    win_df = pd.DataFrame(win_data).set_index('Team')
    fig, ax = plt.subplots(figsize=(15, 8))
    win_df.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    
    plt.title('Team Performance by Victory Type')
    plt.xlabel('Team')
    plt.ylabel('Number of Wins')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels to bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}", 
                       (p.get_x() + p.get_width()/2., height), 
                       ha='center', va='center', 
                       xytext=(0, 5), 
                       textcoords='offset points')
    
    plt.tight_layout()
    plt.show()

def compute_tolerance_interval(data, coverage=0.95, conf=0.95):
    """
    Returns (lower_bound, upper_bound, k_factor) for a two-sided normal tolerance interval.
    
    Parameters:
    - data: Array-like object containing the data
    - coverage: Desired coverage probability (default: 0.95)
    - conf: Confidence level (default: 0.95)
    
    Returns:
    - Tuple of (lower_bound, upper_bound, k_factor)
    """
    n = len(data)
    mean = data.mean()
    std = data.std(ddof=1)
    # chi-square quantile for df = n-1
    chi2_val = chi2.ppf(conf, df=n-1)
    # k factor (see statistical tables or NIST formula)
    k = np.sqrt((n-1) * chi2_val / (n * (n-1)))
    lower = mean - k * std
    upper = mean + k * std
    return lower, upper, k

def perform_statistical_analysis(df_runs):
    """
    Performs statistical analysis including confidence intervals, hypothesis testing,
    and tolerance intervals for each format.
    
    Parameters:
    - df_runs: Dataframe containing runs-based match data
    """
    # Split data into training and test sets
    train_data = df_runs.sample(frac=0.8, random_state=42)
    test_data = df_runs.drop(train_data.index)
    
    # Calculate confidence intervals for mean margin
    n = len(train_data)
    train_mean = train_data['Margin'].mean()
    train_std = train_data['Margin'].std(ddof=1)
    t_crit = stats.t.ppf(0.975, n-1)
    ci_mean = (train_mean - t_crit*train_std/np.sqrt(n), 
              train_mean + t_crit*train_std/np.sqrt(n))
    
    print("\n=== Overall Statistical Intervals ===")
    print(f"95% Confidence Interval (Mean): ({ci_mean[0]:.2f}, {ci_mean[1]:.2f})")
    
    # Compute overall tolerance interval
    ti_lower, ti_upper, k = compute_tolerance_interval(train_data['Margin'])
    coverage_frac = ((test_data['Margin'] >= ti_lower) & 
                    (test_data['Margin'] <= ti_upper)).mean() * 100
    
    print("\n=== Overall Tolerance Interval (95/95) ===")
    print(f"k-factor: {k:.4f}")
    print(f"Interval: ({ti_lower:.2f}, {ti_upper:.2f}) runs")
    print(f"Test-set coverage: {coverage_frac:.2f}%")
    
    # Format-specific analysis
    print("\n=== Format-Specific Tolerance Intervals (95/95) ===")
    for format_name in df_runs['Format'].unique():
        format_train = train_data[train_data['Format'] == format_name]
        format_test = test_data[test_data['Format'] == format_name]
        
        if len(format_train) > 0 and len(format_test) > 0:
            # Compute format-specific tolerance interval
            ti_lower, ti_upper, k = compute_tolerance_interval(format_train['Margin'])
            coverage_frac = ((format_test['Margin'] >= ti_lower) & 
                           (format_test['Margin'] <= ti_upper)).mean() * 100
            
            print(f"\n{format_name}:")
            print(f"k-factor: {k:.4f}")
            print(f"Interval: ({ti_lower:.2f}, {ti_upper:.2f}) runs")
            print(f"Test-set coverage: {coverage_frac:.2f}%")
            print(f"Sample size (train/test): {len(format_train)}/{len(format_test)}")
    
    # Perform home advantage analysis
    if 'Ground' in df_runs.columns:
        # Prepare data for home advantage test
        valid_matches = df_runs.dropna(subset=['Team 1', 'Winner']).copy()
        valid_matches['Home_Win'] = (valid_matches['Winner'] == valid_matches['Team 1']).astype(int)
        
        # Calculate home win statistics
        home_wins = valid_matches['Home_Win'].sum()
        total_matches = len(valid_matches)
        
        # Perform binomial test for home advantage
        result = stats.binomtest(home_wins, total_matches, 0.5, alternative='greater')
        
        print("\n=== Home Advantage Analysis ===")
        print(f"Home Win Rate: {home_wins/total_matches:.2%}")
        print(f"P-value: {result.pvalue:.25f}")
        print("Conclusion:", "Significant home advantage" if result.pvalue < 0.05 else "No significant advantage")

def main():
    """
    Main function that orchestrates the entire analysis process.
    """
    # Try default file path
    default_path = 'Cricket-all-teams-all-matches.csv'
    if not os.path.isfile(default_path):
        print(f"File '{default_path}' not found.")
        file_path = input("Please enter the full path to the cricket CSV file: ").strip()
    else:
        file_path = default_path

    # Load and preprocess data
    df, df_runs = load_and_preprocess_data(file_path)
    
    # Parse match margins
    df['Margin_Value'], df['Outcome_Type'] = zip(*df['Margin'].apply(parse_margin))
    
    # Calculate and display format statistics
    format_stats = calculate_format_statistics(df_runs)
    print("\n=== Comparative Format Statistics (Runs Only) ===")
    print(format_stats.round(2))
    
    # Create visualizations
    plot_match_outcomes(df)
    plot_margin_histograms(df)
    analyze_team_performance(df)
    
    # Perform statistical analysis
    perform_statistical_analysis(df_runs)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()