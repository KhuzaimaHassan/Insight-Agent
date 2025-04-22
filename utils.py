import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tempfile
import os
from fpdf import FPDF
import json
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

def read_file(uploaded_file):
    """Read uploaded file (CSV or Excel) into a pandas DataFrame"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
    return df

def get_data_profile(df):
    """Generate basic data profile"""
    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values_pct": round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numerical_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist(),
    }
    return profile

def clean_data(df, method='drop', fill_value=None, columns=None):
    """Clean data based on method"""
    if columns is None:
        columns = df.columns
    
    if method == 'drop':
        df = df.dropna(subset=columns)
    elif method == 'mean':
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
    elif method == 'median':
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
    elif method == 'mode':
        for col in columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    elif method == 'value':
        df[columns] = df[columns].fillna(fill_value)
    
    return df

def normalize_data(df, columns=None, method='minmax'):
    """Normalize numerical data"""
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        scaler = MinMaxScaler()
        df_normalized[columns] = scaler.fit_transform(df[columns])
    elif method == 'standard':
        scaler = StandardScaler()
        df_normalized[columns] = scaler.fit_transform(df[columns])
    
    return df_normalized

def get_automatic_visualizations(df):
    """Generate automatic visualizations based on data types"""
    visualizations = []
    
    # For numerical columns
    num_cols = df.select_dtypes(include=['number']).columns
    
    # Sample up to 5 numerical columns for visualization
    for col in num_cols[:5]:
        # Histogram
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        visualizations.append(fig)
        
        # Check if there's a potential time series column
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if date_cols and len(date_cols) > 0:
            try:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                fig = px.line(df, x=date_cols[0], y=col, title=f"{col} Over Time")
                visualizations.append(fig)
            except:
                pass
    
    # For categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Sample up to 5 categorical columns for visualization
    for col in cat_cols[:5]:
        if df[col].nunique() < 15:  # Only if there aren't too many unique values
            # Bar chart
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, 'count']
            fig = px.bar(value_counts, x=col, y='count', title=f"Count by {col}")
            visualizations.append(fig)
            
            # If we have numerical columns, create a relationship chart
            if len(num_cols) > 0:
                # Box plot
                fig = px.box(df, x=col, y=num_cols[0], title=f"{num_cols[0]} by {col}")
                visualizations.append(fig)
    
    # If we have at least 2 numerical columns, create a scatter plot
    if len(num_cols) >= 2:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1], title=f"{num_cols[1]} vs {num_cols[0]}")
        visualizations.append(fig)
    
    return visualizations

def create_report(df, profile, visualizations, insights, executive_summary="", recommendations=[], title="Data Analysis Report"):
    """Create a report in HTML format"""
    # Convert insights to HTML if it's a list
    if isinstance(insights, list):
        insights_html = ""
        for insight in insights:
            insights_html += f'<div class="insight-card"><p>{insight}</p></div>'
    else:
        insights_html = insights
    
    # Convert recommendations to HTML if available
    recommendations_html = ""
    if recommendations:
        for rec in recommendations:
            recommendations_html += f'<div class="recommendation-card"><p>💡 {rec}</p></div>'
    
    html_report = f"""
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2C3E50;
            }}
            .header {{
                text-align: center;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            .section {{
                margin-bottom: 40px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .insight-card {{
                background-color: #f8f9fa;
                border-left: 4px solid #4285f4;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 0 4px 4px 0;
            }}
            .recommendation-card {{
                background-color: #e8f5e9;
                border-left: 4px solid #4caf50;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 0 4px 4px 0;
            }}
            .summary-card {{
                background-color: #e3f2fd;
                border: 1px solid #bbdefb;
                padding: 20px;
                margin-bottom: 25px;
                border-radius: 8px;
            }}
            .visualization {{
                margin-bottom: 30px;
                text-align: center;
            }}
            .footer {{
                text-align: center;
                font-size: 0.9rem;
                color: #666;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }}
            .data-metrics {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
            }}
            .metric-card {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                width: 22%;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 1.8rem;
                font-weight: bold;
                color: #1976d2;
                margin: 10px 0;
            }}
            .metric-label {{
                font-size: 0.9rem;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </div>
        
        <!-- Executive Summary Section -->
        {f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-card">
                <p>{executive_summary}</p>
            </div>
        </div>
        """ if executive_summary else ""}
        
        <!-- Dataset Overview Section -->
        <div class="section">
            <h2>Dataset Overview</h2>
            
            <div class="data-metrics">
                <div class="metric-card">
                    <div class="metric-label">Rows</div>
                    <div class="metric-value">{profile['rows']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Columns</div>
                    <div class="metric-value">{profile['columns']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Missing Values</div>
                    <div class="metric-value">{profile['missing_values_pct']}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Data Types</div>
                    <div class="metric-value">{len(profile['numerical_columns'])}/{len(profile['categorical_columns'])}</div>
                    <div class="metric-label">Num/Cat</div>
                </div>
            </div>
            
            <h3>Data Sample</h3>
            {df.head(5).to_html(index=False, classes="table table-striped")}
        </div>
        
        <!-- Data Profile Section -->
        <div class="section">
            <h2>Data Profile</h2>
            
            <h3>Column Information</h3>
            {pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Missing': df.isna().sum(),
                'Missing %': (df.isna().sum() / len(df) * 100).round(2)
            }).to_html(index=False, classes="table table-striped")}
            
            <h3>Numerical Columns</h3>
            <p>{', '.join(profile['numerical_columns'])}</p>
            
            <h3>Categorical Columns</h3>
            <p>{', '.join(profile['categorical_columns'])}</p>
            
            <h3>Summary Statistics</h3>
            {df.describe().to_html(classes="table table-striped")}
        </div>
        
        <!-- Key Insights Section -->
        {f"""
        <div class="section">
            <h2>Key Insights</h2>
            {insights_html}
        </div>
        """ if insights_html else ""}
        
        <!-- Recommendations Section -->
        {f"""
        <div class="section">
            <h2>Recommendations</h2>
            {recommendations_html}
        </div>
        """ if recommendations_html else ""}
        
        <div class="footer">
            <p>Report generated using AI-powered Data Analysis</p>
        </div>
    </body>
    </html>
    """
    
    return html_report

def plot_time_series(df, x_col, y_col, title=None):
    """Create a time series plot"""
    if title is None:
        title = f"{y_col} Over Time"
    
    fig = px.line(df, x=x_col, y=y_col, title=title)
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white"
    )
    return fig

def plot_bar_chart(df, x_col, y_col=None, title=None):
    """Create a bar chart"""
    if y_col is None:
        # Count by category
        value_counts = df[x_col].value_counts().reset_index()
        value_counts.columns = [x_col, 'count']
        fig = px.bar(value_counts, x=x_col, y='count', title=f"Count by {x_col}" if title is None else title)
    else:
        # Aggregation
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}" if title is None else title)
    
    fig.update_layout(template="plotly_white")
    return fig

def process_natural_language_query(query, df):
    """Simple rule-based query processor"""
    query = query.lower()
    result = {"type": None, "data": None, "message": ""}
    
    # Extract columns mentioned in the query
    columns = [col for col in df.columns if col.lower() in query]
    
    if not columns:
        return {"type": "error", "message": "Couldn't identify any columns in your query. Please be more specific."}
    
    # Check for common operations
    if "average" in query or "mean" in query:
        # Find numerical columns from the mentioned columns
        num_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        if num_cols:
            result["type"] = "statistic"
            result["data"] = {col: df[col].mean() for col in num_cols}
            result["message"] = f"Average of {', '.join(num_cols)}"
    
    elif "show" in query and "by" in query:
        # This is probably a groupby operation
        # Find the columns for groupby and aggregation
        cols_mentioned = [col for col in df.columns if col.lower() in query]
        if len(cols_mentioned) >= 2:
            group_col = cols_mentioned[0]
            value_col = cols_mentioned[1]
            
            if pd.api.types.is_numeric_dtype(df[value_col]):
                # Create a grouped bar chart
                result["type"] = "visualization"
                grouped_data = df.groupby(group_col)[value_col].mean().reset_index()
                fig = px.bar(grouped_data, x=group_col, y=value_col, 
                             title=f"{value_col} by {group_col}")
                result["data"] = fig
                result["message"] = f"Showing {value_col} by {group_col}"
    
    elif any(word in query for word in ["where", "most", "highest", "top"]):
        # This is probably a filtering/ranking operation
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols and "highest" in query or "top" in query:
            # Sort by a numeric column and return top values
            col = numeric_cols[0]
            top_data = df.sort_values(by=col, ascending=False).head(5)
            result["type"] = "data"
            result["data"] = top_data
            result["message"] = f"Showing top values by {col}"
        
        elif categorical_cols:
            # Count by category and find the highest
            col = categorical_cols[0]
            counts = df[col].value_counts().head(5)
            result["type"] = "data"
            result["data"] = counts
            result["message"] = f"Showing counts for {col}"
    
    # If no specific operation was detected but columns were found
    if result["type"] is None and columns:
        col = columns[0]
        if pd.api.types.is_numeric_dtype(df[col]):
            # Show distribution for numeric column
            result["type"] = "visualization"
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            result["data"] = fig
            result["message"] = f"Showing distribution of {col}"
        else:
            # Show counts for categorical column
            result["type"] = "visualization"
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, 'count']
            fig = px.bar(counts, x=col, y='count', title=f"Count by {col}")
            result["data"] = fig
            result["message"] = f"Showing counts for {col}"
    
    return result

def get_automatic_insights(df):
    """Generate basic insights automatically"""
    insights = []
    
    # Check for missing values
    missing_percentage = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    if missing_percentage > 0:
        insights.append(f"Dataset contains {missing_percentage:.2f}% missing values.")
    
    # Check for numerical columns
    num_cols = df.select_dtypes(include=['number']).columns
    
    for col in num_cols:
        # Check for skewness
        skewness = df[col].skew()
        if abs(skewness) > 1:
            direction = "right" if skewness > 0 else "left"
            insights.append(f"{col} is significantly skewed to the {direction} (skewness: {skewness:.2f}).")
        
        # Check for outliers using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
        if len(outliers) > 0:
            insights.append(f"{col} has {len(outliers)} potential outliers ({(len(outliers)/len(df)*100):.2f}% of data).")
    
    # Check for categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        # Check cardinality
        unique_count = df[col].nunique()
        if unique_count == 1:
            insights.append(f"{col} has only one unique value: '{df[col].iloc[0]}'.")
        elif unique_count < 5:
            insights.append(f"{col} has low cardinality with only {unique_count} unique values.")
        
        # Check for imbalance in categories
        value_counts = df[col].value_counts(normalize=True)
        if value_counts.iloc[0] > 0.8:  # If top category is more than 80%
            insights.append(f"{col} is imbalanced with '{value_counts.index[0]}' representing {value_counts.iloc[0]*100:.2f}% of values.")
    
    # Check for time series patterns if date columns exist
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    for date_col in date_cols:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            if df[date_col].min().year != df[date_col].max().year:
                years_span = df[date_col].max().year - df[date_col].min().year
                insights.append(f"Data spans {years_span} years, from {df[date_col].min().year} to {df[date_col].max().year}.")
        except:
            pass
    
    # Check relationships between numeric columns
    if len(num_cols) >= 2:
        for i, col1 in enumerate(num_cols):
            for col2 in num_cols[i+1:]:
                corr = df[col1].corr(df[col2])
                if abs(corr) > 0.7:
                    direction = "positive" if corr > 0 else "negative"
                    insights.append(f"Strong {direction} correlation ({corr:.2f}) between {col1} and {col2}.")
    
    return insights

def fix_dataframe_for_arrow(df):
    """
    Fix DataFrame to make it compatible with PyArrow by converting problematic columns.
    This addresses the "Could not convert dtype('O')" error.
    """
    if df is None:
        return None
    
    # Create a copy to avoid modifying the original dataframe
    df_fixed = df.copy()
    
    # Convert object dtype columns to string dtype
    for col in df_fixed.select_dtypes(include=['object']).columns:
        try:
            df_fixed[col] = df_fixed[col].astype(str)
        except Exception as e:
            print(f"Could not convert column {col}: {str(e)}")
    
    return df_fixed