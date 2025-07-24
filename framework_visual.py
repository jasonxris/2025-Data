#!/usr/bin/env python3
"""
--------------------------------------------------------------------
framework_visual.py
--------------------------------------------------------------------
Creates a 2x2 Framework quadrant visualization using Plotly Express
to classify trades by Risk Appetite and Holding Period.

Quadrants:
- Quick Skims (Low-Risk, Short-Hold): small, fast chops
- Speculative Flips (High-Risk, Short-Hold): big, fast punts  
- Strategic Core (Low-Risk, Long-Hold): small, patient positions
- High-Conviction Bets (High-Risk, Long-Hold): large, patient stakes

X-axis: Holding Period (Short-Hold < 8 days, Long-Hold ≥ 8 days)
Y-axis: Risk Level (Low-Risk < 8%, High-Risk ≥ 8% of $1M portfolio)
Bubble size: Realized P/L ($)
Color: Highlights "Seventh Sense Investing" vs peers (grey)
--------------------------------------------------------------------
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path


# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
DATA_FILE = Path(__file__).resolve().parent / "output" / "processed_trades.csv"
TARGET_TEAM = "Seventh Sense Investing"  # Team to highlight in color
OUTPUT_FILE = Path(__file__).resolve().parent / "output" / "framework_visual.html"


# -----------------------------------------------------------------
# Load and prepare data
# -----------------------------------------------------------------
def load_trades_data():
    """Load processed trades and prepare for visualization"""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Please run build_processed_trades.py first. Missing: {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)
    
    # Debug: Print unique team names to verify matching
    print(f"Available teams: {df['team'].unique()}")
    print(f"Looking for target team: '{TARGET_TEAM}'")
    print(f"Target team found: {TARGET_TEAM in df['team'].values}")
    
    # Filter out extreme outliers above 300% portfolio
    df = df[df['pct_portfolio'] <= 300.0].copy()
    print(f"Filtered to {len(df)} trades (removed trades > 300% portfolio)")
    
    # Create log-transformed axes for balanced quadrants
    # Use log10(x + 1) to handle values close to 0
    df['log_pct_portfolio'] = np.log10(df['pct_portfolio'] + 1)
    df['log_holding_days'] = np.log10(df['holding_days'] + 1)
    
    # Calculate log thresholds for quadrant dividers
    log_risk_threshold = np.log10(8.0 + 1)  # 8% portfolio
    log_hold_threshold = np.log10(8 + 1)    # 8 days
    
    # Create quadrant labels based on actual values (using original pct_portfolio)
    def get_quadrant(row):
        if row['pct_portfolio'] < 8.0 and row['holding_days'] < 8:
            return 'Quick Skims<br>(small, fast chops)'
        elif row['pct_portfolio'] >= 8.0 and row['holding_days'] < 8:
            return 'Speculative Flips<br>(big, fast punts)'
        elif row['pct_portfolio'] < 8.0 and row['holding_days'] >= 8:
            return 'Strategic Core<br>(small, patient positions)'
        else:  # High-Risk, Long-Hold
            return 'High-Conviction Bets<br>(large, patient stakes)'
    
    df['quadrant'] = df.apply(get_quadrant, axis=1)
    
    # Color mapping based on profit/loss
    df['profit_color'] = df['trade_result'].map({'Positive': 'green', 'Negative': 'red'})
    
    # Team identification for opacity and outline
    df['is_target_team'] = df['team'] == TARGET_TEAM
    
    # Absolute P/L for bubble sizing with much larger minimum size
    df['abs_pl'] = df['total_pl_usd'].abs()
    df['pl_size'] = df['abs_pl'] + 30000  # 4x larger minimum size for better visibility
    
    return df, log_risk_threshold, log_hold_threshold


# -----------------------------------------------------------------
# Create visualization
# -----------------------------------------------------------------
def create_framework_chart(df, log_risk_threshold, log_hold_threshold):
    """Create the 2x2 Framework quadrant visualization"""
    
    # Create scatter plot using px.scatter with profit/loss colors
    fig = px.scatter(
        df,
        x='log_holding_days',
        y='log_pct_portfolio',
        size='pl_size',
        color='profit_color',
        color_discrete_map={'green': 'green', 'red': 'red'},
        hover_data={
            'team': True,
            'symbol': True,
            'total_pl_usd': ':$,.0f',
            'pct_portfolio': ':.2f%',
            'holding_days': True,
            'trade_result': True,
            'quadrant': True,
            'pl_size': False,
            'profit_color': False,
            'is_target_team': False,
            'log_pct_portfolio': False,
            'log_holding_days': False
        },
        title='Trading Framework: Risk Appetite vs Holding Period Analysis (Log Scale)',
        width=800,
        height=600
    )
    
    # Apply styling per individual point within each trace
    for i, trace in enumerate(fig.data):
        if hasattr(trace, 'customdata') and len(trace.customdata) > 0:
            # Get team names for this trace
            team_names = trace.customdata[:, 0]  # Team names are in first column
            
            # Create arrays for opacity and outline color based on team
            opacity_array = [1.0 if team == TARGET_TEAM else 0.3 for team in team_names]
            outline_array = ['rgba(0,0,0,0.4)' if team == TARGET_TEAM else 'rgba(0,0,0,0)' for team in team_names]
            
            # Apply the arrays to this trace
            trace.marker.opacity = opacity_array
            trace.marker.line.color = outline_array
            trace.marker.line.width = 1
            
            print(f"Trace {i}: Applied individual point styling to {len(team_names)} points")
    
    # Create custom axis labels for log scales
    y_log_min = df['log_pct_portfolio'].min()
    y_log_max = df['log_pct_portfolio'].max()
    x_log_min = df['log_holding_days'].min()
    x_log_max = df['log_holding_days'].max()
    
    # Create y-axis tick positions and labels (portfolio %)
    y_tick_values = [np.log10(1), np.log10(2), np.log10(4), log_risk_threshold, np.log10(16), np.log10(50), np.log10(100), np.log10(200)]
    y_tick_labels = ['1%', '2%', '4%', '8%', '16%', '50%', '100%', '200%']
    
    # Create x-axis tick positions and labels (holding days)
    x_tick_values = [np.log10(1), np.log10(2), np.log10(4), log_hold_threshold, np.log10(16), np.log10(30), np.log10(60), np.log10(120)]
    x_tick_labels = ['1', '2', '4', '8', '16', '30', '60', '120']
    
    # Filter ticks to reasonable ranges
    valid_y_ticks = [(val, label) for val, label in zip(y_tick_values, y_tick_labels) 
                     if y_log_min <= val <= y_log_max]
    valid_x_ticks = [(val, label) for val, label in zip(x_tick_values, x_tick_labels) 
                     if x_log_min <= val <= x_log_max]
    
    # Customize layout
    fig.update_layout(
        xaxis=dict(
            title='Holding Period (Days) - Log Scale',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            tickmode='array',
            tickvals=[val for val, _ in valid_x_ticks],
            ticktext=[label for _, label in valid_x_ticks]
        ),
        yaxis=dict(
            title='Risk Level (% of $1M Portfolio) - Log Scale',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            tickmode='array',
            tickvals=[val for val, _ in valid_y_ticks],
            ticktext=[label for _, label in valid_y_ticks]
        ),
        legend=dict(
            title='Team',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    # Calculate midpoints for quadrant labels in log space
    x_log_mid_short = (x_log_min + log_hold_threshold) / 2  # Midpoint of 0-8 days range
    x_log_mid_long = (log_hold_threshold + x_log_max) / 2   # Midpoint of 8+ days range
    y_log_mid_low = (y_log_min + log_risk_threshold) / 2   # Midpoint of 0-8% range
    y_log_mid_high = (log_risk_threshold + y_log_max) / 2  # Midpoint of 8%+ range
    
    # Add quadrant labels positioned based on log-transformed ranges
    quadrant_labels = [
        dict(x=x_log_mid_short, y=y_log_mid_low, text='Quick Skims<br><i>(small, fast chops)</i>', showarrow=False, 
             font=dict(size=12, color='gray'), xanchor='center', yanchor='middle'),
        dict(x=x_log_mid_long, y=y_log_mid_low, text='Strategic Core<br><i>(small, patient positions)</i>', showarrow=False,
             font=dict(size=12, color='gray'), xanchor='center', yanchor='middle'),
        dict(x=x_log_mid_short, y=y_log_mid_high, text='Speculative Flips<br><i>(big, fast punts)</i>', showarrow=False,
             font=dict(size=12, color='gray'), xanchor='center', yanchor='middle'),
        dict(x=x_log_mid_long, y=y_log_mid_high, text='High-Conviction Bets<br><i>(large, patient stakes)</i>', showarrow=False,
             font=dict(size=12, color='gray'), xanchor='center', yanchor='middle')
    ]
    
    fig.update_layout(annotations=quadrant_labels)
    
    # Add dividing lines at threshold values
    fig.add_hline(y=log_risk_threshold, line_dash="dash", line_color="black", opacity=0.7, 
                  annotation_text="8% Portfolio Threshold", annotation_position="right")
    fig.add_vline(x=log_hold_threshold, line_dash="dash", line_color="black", opacity=0.7,
                  annotation_text="8 Day Threshold", annotation_position="top")
    
    return fig


# -----------------------------------------------------------------
# Generate summary stats
# -----------------------------------------------------------------
def generate_summary(df):
    """Generate summary statistics by quadrant"""
    target_df = df[df['team'] == TARGET_TEAM] if TARGET_TEAM in df['team'].values else df
    
    print(f"\n=== FRAMEWORK ANALYSIS: {TARGET_TEAM if TARGET_TEAM in df['team'].values else 'ALL TEAMS'} ===")
    print(f"Total Trades: {len(target_df)}")
    print(f"Total P/L: ${target_df['total_pl_usd'].sum():,.0f}")
    print(f"Win Rate: {(target_df['trade_result'] == 'Positive').mean():.1%}")
    
    print(f"\n--- Quadrant Breakdown ---")
    quadrant_stats = target_df.groupby(['risk_level', 'hold_type']).agg({
        'total_pl_usd': ['count', 'sum', 'mean'],
        'pct_portfolio': 'mean'
    }).round(2)
    
    for (risk, hold), group in target_df.groupby(['risk_level', 'hold_type']):
        quadrant_name = get_quadrant_name(risk, hold)
        trades = len(group)
        total_pl = group['total_pl_usd'].sum()
        avg_pl = group['total_pl_usd'].mean()
        avg_size = group['pct_portfolio'].mean()
        
        print(f"{quadrant_name}: {trades} trades, ${total_pl:,.0f} total P/L, "
              f"${avg_pl:,.0f} avg P/L, {avg_size:.1f}% avg size")


def get_quadrant_name(risk, hold):
    """Get quadrant name from risk and hold values"""
    if risk == 'Low-Risk' and hold == 'Short-Hold':
        return 'Quick Skims'
    elif risk == 'High-Risk' and hold == 'Short-Hold':
        return 'Speculative Flips'
    elif risk == 'Low-Risk' and hold == 'Long-Hold':
        return 'Strategic Core'
    else:
        return 'High-Conviction Bets'


# -----------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------
def main():
    """Generate Framework visualization and summary"""
    try:
        # Load data
        df, log_risk_threshold, log_hold_threshold = load_trades_data()
        print(f"Loaded {len(df)} trades from {DATA_FILE}")
        
        # Create visualization
        fig = create_framework_chart(df, log_risk_threshold, log_hold_threshold)
        
        # Save interactive HTML
        fig.write_html(OUTPUT_FILE)
        print(f"Framework visualization saved to: {OUTPUT_FILE}")
        
        # Show in browser (optional - comment out if not desired)
        fig.show()
        
        # Generate summary statistics
        generate_summary(df)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run build_processed_trades.py first to generate the data file.")


if __name__ == "__main__":
    main()