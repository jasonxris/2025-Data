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

X-axis: Holding Period (Short-Hold < 6 days, Long-Hold ≥ 6 days)
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
    
    # Debug: Print basic info
    print(f"Target team found: {TARGET_TEAM in df['team'].values}")
    
    print(f"Loaded {len(df)} trades from data file")
    
    # Create evenly-spaced scaled y-axis positions to match tick system
    def even_portfolio_scale(pct):
        """Scale portfolio percentages to match evenly-spaced tick system"""
        # Define the percentage breakpoints including 0% and 300%
        y_percentages = [0, 1, 2, 4, 8, 16, 50, 100, 200, 300]
        
        # Find which segment the percentage falls into
        if pct <= y_percentages[0]:  # At or below 0%
            return 0.0
        elif pct >= y_percentages[-1]:  # Above 300%
            return len(y_percentages) - 1
        else:
            # Find the segment and interpolate
            for i in range(len(y_percentages) - 1):
                if y_percentages[i] <= pct <= y_percentages[i + 1]:
                    # Linear interpolation between the two tick positions
                    ratio = (pct - y_percentages[i]) / (y_percentages[i + 1] - y_percentages[i])
                    return i + ratio
        
        return 0.0  # Fallback
    
    df['log_pct_portfolio'] = df['pct_portfolio'].apply(even_portfolio_scale)
    
    # Add jitter to spread out same-day (0 day) trades only
    np.random.seed(42)  # For reproducible results
    df['holding_days_jittered'] = df['holding_days'].astype(float).copy()
    
    # For same-day trades (0 days), add jitter between 0-1 to spread them out
    same_day_mask = df['holding_days'] == 0
    jitter_amount = np.random.uniform(0.1, 0.9, size=same_day_mask.sum())
    df.loc[same_day_mask, 'holding_days_jittered'] = jitter_amount
    
    df['log_holding_days'] = np.log10(df['holding_days_jittered'] + 1)
    
    
    # Calculate log thresholds for quadrant dividers
    log_hold_threshold = np.log10(6 + 1)    # 6 days (using original value, not jittered)
    
    # Create quadrant labels based on actual values (using original pct_portfolio)
    def get_quadrant(row):
        if row['pct_portfolio'] < 8.0 and row['holding_days'] < 6:
            return 'Quick Skims<br>(small, fast chops)'
        elif row['pct_portfolio'] >= 8.0 and row['holding_days'] < 6:
            return 'Speculative Flips<br>(big, fast punts)'
        elif row['pct_portfolio'] < 8.0 and row['holding_days'] >= 6:
            return 'Strategic Core<br>(small, patient positions)'
        else:  # High-Risk, Long-Hold
            return 'High-Conviction Bets<br>(large, patient stakes)'
    
    df['quadrant'] = df.apply(get_quadrant, axis=1)
    
    # Color mapping based on profit/loss
    df['profit_color'] = df['trade_result'].map({'Positive': 'green', 'Negative': 'red'})
    
    # Create user-friendly trade result labels
    df['trade_result_display'] = df['trade_result'].map({'Positive': 'Gain', 'Negative': 'Loss'})
    
    # Team identification for opacity and outline
    df['is_target_team'] = df['team'] == TARGET_TEAM
    
    # Sort so target team dots are drawn last (on top)
    df = df.sort_values('is_target_team', ascending=True)  # False first, True last
    
    # Better bubble sizing with proper scaling
    df['abs_pl'] = df['total_pl_usd'].abs()
    
    # Define size range for bubbles - much wider range for better differentiation
    min_bubble_size = 5000   # Very small for tiny trades
    max_bubble_size = 300000  # Large for big trades
    
    # Get the actual P/L range in your data
    min_pl = df['abs_pl'].min()
    max_pl = df['abs_pl'].max()
    
    print(f"P/L range: ${min_pl:,.0f} to ${max_pl:,.0f}")
    
    # Much more aggressive scaling with multiple breakpoints
    def scale_bubble_size(pl_value):
        if max_pl == min_pl:  # Avoid division by zero
            return min_bubble_size
        
        # Create breakpoints for different scaling behavior
        if pl_value <= 1000:  # Very small trades
            # Map $0-$1000 to size 5000-15000
            ratio = pl_value / 1000
            return 5000 + 10000 * ratio
            
        elif pl_value <= 25000:  # Small-medium trades  
            # Map $1000-$25000 to size 15000-80000 (big jump here)
            ratio = (pl_value - 1000) / (25000 - 1000)
            return 15000 + 65000 * ratio
            
        elif pl_value <= 100000:  # Medium-large trades
            # Map $25000-$100000 to size 80000-150000
            ratio = (pl_value - 25000) / (100000 - 25000)
            return 80000 + 70000 * ratio
            
        else:  # Very large trades
            # Map $100000+ to size 150000-300000
            ratio = min(1.0, (pl_value - 100000) / (max_pl - 100000))
            return 150000 + 150000 * ratio
    
    df['pl_size'] = df['abs_pl'].apply(scale_bubble_size)
    
    return df, log_hold_threshold


# -----------------------------------------------------------------
# Create visualization
# -----------------------------------------------------------------
def create_framework_chart(df, log_hold_threshold):
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
            'trade_result_display': True,
            'pl_size': False,
            'profit_color': False,
            'is_target_team': False,
            'log_pct_portfolio': False,
            'log_holding_days': False
        },
        title='Risk Appetite vs Holding Period Analysis',
        width=800,
        height=600
    )
    
    # Make all dots 1.75x larger by adjusting sizeref
    fig.update_traces(marker=dict(sizeref=2.*max(df['pl_size'])/(33.**2)/1.01))
    
    # Apply styling per individual point within each trace
    for i, trace in enumerate(fig.data):
        if hasattr(trace, 'customdata') and len(trace.customdata) > 0:
            # Get team names for this trace
            team_names = trace.customdata[:, 0]  # Team names are in first column
            
            # Create arrays for opacity and outline color based on team
            opacity_array = [1.0 if team == TARGET_TEAM else 0.3 for team in team_names]
            outline_array = ['rgba(0,0,0,0.6)' if team == TARGET_TEAM else 'rgba(0,0,0,0)' for team in team_names]
            
            # Apply the arrays to this trace
            trace.marker.opacity = opacity_array
            trace.marker.line.color = outline_array
            trace.marker.line.width = 1.2  # Bolder outline
            
            # Update hover template with better labels
            trace.hovertemplate = (
                '<b>%{customdata[0]}</b><br>' +
                'Stock: %{customdata[1]}<br>' +
                'Total Profit/Loss: %{customdata[2]:$,.0f}<br>' +
                'Percent of Portfolio: %{customdata[3]:.2f}%<br>' +
                'Holding Days: %{customdata[4]}<br>' +
                'Trade Result: %{customdata[5]}<br>' +
                '<extra></extra>'
            )
            
    
    # Get axis ranges
    y_log_min = df['log_pct_portfolio'].min()
    y_log_max = df['log_pct_portfolio'].max()
    x_log_min = df['log_holding_days'].min()
    x_log_max = df['log_holding_days'].max()
    
    # Create evenly spaced y-axis ticks that represent the correct percentage values
    y_percentages = [0, 1, 2, 4, 8, 16, 50, 100, 200, 300]
    y_tick_labels = [f'{p}%' for p in y_percentages]
    
    # Create evenly spaced positions - use index positions directly
    y_tick_values = list(range(len(y_percentages)))
    
    # Calculate the position of 8% threshold in the evenly spaced system
    # 8% is at index 4 in the y_percentages list [0, 1, 2, 4, 8, 16, 50, 100, 200, 300]
    log_risk_threshold = 4
    
    # Create x-axis tick positions and labels (holding days) - using +1 for log transform consistency
    x_tick_values = [np.log10(1+1), np.log10(2+1), np.log10(4+1), log_hold_threshold, np.log10(16+1), np.log10(30+1), np.log10(60+1), np.log10(120+1)]
    x_tick_labels = ['1', '2', '4', '6', '16', '30', '60', '120']
    
    # Filter x-axis ticks to reasonable ranges
    valid_x_ticks = [(val, label) for val, label in zip(x_tick_values, x_tick_labels) 
                     if x_log_min <= val <= x_log_max]
    
    # Customize layout
    fig.update_layout(
        xaxis=dict(
            title='Holding Period (Days)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            tickmode='array',
            tickvals=[val for val, _ in valid_x_ticks],
            ticktext=[label for _, label in valid_x_ticks],
            range=[x_log_min - 0.1, x_log_max + 0.3]  # Add extra buffer on the right
        ),
        yaxis=dict(
            title='Risk Level (% of $1M Portfolio)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            tickmode='array',
            tickvals=y_tick_values,
            ticktext=y_tick_labels,
            range=[-2.0, len(y_percentages) - 1 + 2.0]  # Extra buffer space above and below for manual quadrant stats
        ),
        showlegend=False,  # Hide the automatic color legend
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    # Calculate midpoints for quadrant labels - use padding areas for y-positioning
    x_log_mid_short = (x_log_min + log_hold_threshold) / 2  # Midpoint of 0-8 days range
    x_log_mid_long = (log_hold_threshold + x_log_max) / 2   # Midpoint of 8+ days range
    
    # Position labels higher up to leave space below them for statistics
    y_label_bottom = -0.5   # Bottom padding area for titles (leave space below for stats)
    y_label_top = len(y_percentages) - 1 + 1.7     # Top padding area for titles (leave space below for stats)
    
    # Simple clean quadrant labels only
    quadrant_labels = [
        dict(x=x_log_mid_short, y=y_label_bottom, text='Quick Skims', showarrow=False, 
             font=dict(size=14, color='black'), xanchor='center', yanchor='middle'),
        dict(x=x_log_mid_long, y=y_label_bottom, text='Strategic Core', showarrow=False,
             font=dict(size=14, color='black'), xanchor='center', yanchor='middle'),
        dict(x=x_log_mid_short, y=y_label_top, text='Speculative Flips', showarrow=False,
             font=dict(size=14, color='black'), xanchor='center', yanchor='middle'),
        dict(x=x_log_mid_long, y=y_label_top, text='High Conviction', showarrow=False,
             font=dict(size=14, color='black'), xanchor='center', yanchor='middle')
    ]
    
    fig.update_layout(annotations=quadrant_labels)
    
    # Add dividing lines at threshold values
    fig.add_hline(y=log_risk_threshold, line_dash="dash", line_color="black", opacity=0.7, 
                  annotation_text="8% Portfolio Threshold", annotation_position="right")
    fig.add_vline(x=log_hold_threshold, line_dash="dash", line_color="black", opacity=0.7,
                  annotation_text="6 Day Threshold", annotation_position="top")
    
    # Add explanatory text annotations for legend in top-right corner
    fig.add_annotation(
        x=0.98, y=0.98, xref="paper", yref="paper",
        text="<b>Color:</b><br>🟢 Green = Gain<br>🔴 Red = Loss<br><br><b>Opacity:</b><br>Bright = Seventh Sense<br>Faded = Other Teams<br><br><b>Size:</b><br>Larger = Higher P/L",
        showarrow=False,
        align="left",
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=10)
    )
    
    # Print quadrant statistics to console
    print("\n" + "="*80)
    print("QUADRANT ANALYSIS")
    print("="*80)
    
    def print_quadrant_stats(quadrant_name, quadrant_filter):
        # All teams data
        all_data = df[quadrant_filter]
        all_trades = len(all_data)
        all_total_pl = all_data['total_pl_usd'].sum()
        all_win_rate = (all_data['total_pl_usd'] > 0).mean() * 100 if len(all_data) > 0 else 0
        
        # Target team data
        target_data = df[(quadrant_filter) & (df['team'] == TARGET_TEAM)]
        target_trades = len(target_data)
        target_total_pl = target_data['total_pl_usd'].sum()
        target_win_rate = (target_data['total_pl_usd'] > 0).mean() * 100 if len(target_data) > 0 else 0
        
        print(f"\n{quadrant_name.upper()}")
        print("-" * 40)
        print(f"Your Team ({TARGET_TEAM}):")
        print(f"  Trades: {target_trades}")
        print(f"  Total P/L: ${target_total_pl:,.2f}")
        print(f"  Win Rate: {target_win_rate:.1f}%")
        print(f"\nAll Teams:")
        print(f"  Trades: {all_trades}")
        print(f"  Total P/L: ${all_total_pl:,.2f}")
        print(f"  Win Rate: {all_win_rate:.1f}%")
    
    # Print stats for each quadrant
    print_quadrant_stats("Quick Skims", (df['pct_portfolio'] < 8.0) & (df['holding_days'] < 6))
    print_quadrant_stats("Strategic Core", (df['pct_portfolio'] < 8.0) & (df['holding_days'] >= 6))
    print_quadrant_stats("Speculative Flips", (df['pct_portfolio'] >= 8.0) & (df['holding_days'] < 6))
    print_quadrant_stats("High-Conviction Bets", (df['pct_portfolio'] >= 8.0) & (df['holding_days'] >= 6))
    
    print("\n" + "="*80)
    
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
        df, log_hold_threshold = load_trades_data()
        print(f"Loaded {len(df)} trades from {DATA_FILE}")
        
        # Create visualization
        fig = create_framework_chart(df, log_hold_threshold)
        
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