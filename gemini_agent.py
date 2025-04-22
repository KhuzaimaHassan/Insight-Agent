import os
import logging
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
    logger.warning("No valid Google API key found in .env file. Please add your API key.")
    # Try to read directly from .env file as fallback
    try:
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                env_contents = f.read()
                if 'GOOGLE_API_KEY=' in env_contents:
                    GOOGLE_API_KEY = env_contents.split('GOOGLE_API_KEY=')[1].split('\n')[0].strip()
                    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    except Exception as e:
        logger.error(f"Error reading API key from .env file: {e}")

class GeminiAgent:
    def __init__(self, model_name="gemini-1.5-flash-latest"):
        """Initialize the agent with Gemini Flash latest model"""
        logger.info(f"Loading Gemini model: {model_name}")
        
        # Flag to track initialization status
        self.initialized = False
        
        try:
            # Configure the API
            if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
                logger.error("No valid Google API key found. Please set your GOOGLE_API_KEY in the .env file.")
                self.model = None
                self.chat = None
                return
                
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Set default safety settings
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # Create the model
            self.model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=self.safety_settings
            )
            
            # Test the model with a simple prompt to verify API key
            try:
                test_response = self.model.generate_content("Hello")
                if not test_response or not hasattr(test_response, 'text'):
                    raise Exception("Model response test failed")
                
                # Initialize chat session
                self.chat = self.model.start_chat(history=[])
                
                # Mark as successfully initialized
                self.initialized = True
                logger.info(f"Successfully loaded Gemini model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to test Gemini model connectivity: {e}")
                self.model = None
                self.chat = None
                
        except Exception as e:
            logger.error(f"Error loading Gemini model: {e}")
            self.model = None
            self.chat = None
    
    def generate_data_summary(self, data_profile, df_sample):
        """Generate a summary of the dataset"""
        if not self.model:
            return "Model not initialized. Please check your API key."
        
        try:
            # Construct a description of the data
            data_desc = f"""
            Dataset: {data_profile['rows']} rows, {data_profile['columns']} columns.
            Missing: {data_profile['missing_values_pct']}%
            Numerical columns: {', '.join(data_profile['numerical_columns'][:5])}
            Categorical columns: {', '.join(data_profile['categorical_columns'][:5])}
            
            Sample data: 
            {df_sample.head(3).to_string()}
            
            Summarize the key characteristics of this dataset in 3-4 sentences.
            """
            
            response = self.model.generate_content(data_desc)
            return response.text
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Could not generate summary. Please check the data format."
    
    def generate_insights(self, df, visualizations, data_profile):
        """Generate concise insights about the dataset"""
        if not self.model:
            return self.get_fallback_insights(df, data_profile)
        
        insights = []
        
        try:
            # Create a more comprehensive description of the dataset
            num_cols = data_profile.get('numerical_columns', [])[:5]  # Include more columns
            cat_cols = data_profile.get('categorical_columns', [])[:5]
            
            # Basic stats for numerical columns
            num_stats = []
            for col in num_cols:
                try:
                    stats = f"{col} (min: {df[col].min():.2f}, max: {df[col].max():.2f}, mean: {df[col].mean():.2f}, median: {df[col].median():.2f})"
                    num_stats.append(stats)
                except:
                    pass
            
            # Basic stats for categorical columns
            cat_stats = []
            for col in cat_cols:
                try:
                    # Get top 2 values for more context
                    top_vals = df[col].value_counts().head(2)
                    stats_parts = []
                    for val, count in top_vals.items():
                        pct = (count / len(df)) * 100
                        stats_parts.append(f"'{val}': {count} ({pct:.1f}%)")
                    
                    stats = f"{col} (top values: {', '.join(stats_parts)})"
                    cat_stats.append(stats)
                except:
                    pass
            
            # Add sample data rows
            sample_data = df.head(5).to_string()
            
            # Create improved prompt for the model
            prompt = f"""
            Generate 4-5 key insights about this dataset:
            
            Dataset: {data_profile['rows']} rows, {data_profile['columns']} columns.
            Missing values: {data_profile['missing_values_pct']}%
            
            All columns: {', '.join(df.columns.tolist())}
            
            Numerical columns with stats:
            {chr(10).join(num_stats)}
            
            Categorical columns with distributions:
            {chr(10).join(cat_stats)}
            
            Sample data (first 5 rows):
            {sample_data}
            
            Each insight should be:
            1. Data-driven and specific with actual values and statistics
            2. Focused on patterns, outliers, or interesting relationships
            3. Clear and direct without vague statements
            4. Actionable where possible
            
            Format each insight as a separate point.
            """
            
            # Get insights from the model
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Split into separate insights
            if '\n' in response_text:
                model_insights = [line.strip() for line in response_text.split('\n') if line.strip()]
            else:
                # Split by sentences if not already formatted
                import re
                model_insights = re.split(r'(?<=[.!?])\s+', response_text)
                model_insights = [insight.strip() for insight in model_insights if insight.strip()][:5]
            
            # Add model-generated insights
            insights.extend(model_insights)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            # Fall back to basic insights
            return self.get_fallback_insights(df, data_profile)
        
        return insights
    
    def answer_question(self, question, df_info):
        """Answer a natural language question about the data"""
        if not self.model:
            return "Model not initialized. Please check your Google API key in the .env file."
        
        try:
            # Check if we received enhanced context format
            if isinstance(df_info, dict) and "full_prompt" in df_info:
                # Use the provided prompt
                prompt = df_info["full_prompt"]
                
                # Add helpful analysis templates
                prompt += """
                
                You can generate and execute code to analyze the data. Here are some helpful analysis templates:
                
                1. For calculating statistics for a column:
                ```python
                # For basic statistics
                df['column_name'].describe()
                
                # For custom calculations
                mean_val = df['column_name'].mean()
                median_val = df['column_name'].median()
                min_val = df['column_name'].min()
                max_val = df['column_name'].max()
                ```
                
                2. For grouping and aggregating data:
                ```python
                # Group by a categorical column and calculate mean of a numerical column
                df.groupby('categorical_column')['numerical_column'].mean()
                
                # Group by multiple columns with multiple aggregations
                df.groupby(['cat_col1', 'cat_col2']).agg({
                    'num_col1': ['mean', 'median'],
                    'num_col2': ['min', 'max', 'count']
                })
                ```
                
                3. For finding top/bottom values:
                ```python
                # Top 5 highest values in a column
                df['column_name'].nlargest(5)
                
                # Bottom 5 values in a column
                df['column_name'].nsmallest(5)
                
                # Top 5 groups by average value
                df.groupby('group_column')['value_column'].mean().nlargest(5)
                ```
                
                4. For correlation analysis:
                ```python
                # Correlation between numerical columns
                df[['num_col1', 'num_col2', 'num_col3']].corr()
                
                # Correlation matrix heatmap
                import seaborn as sns
                sns.heatmap(df[numerical_columns].corr(), annot=True)
                ```
                
                5. For categorical data analysis:
                ```python
                # Frequency counts
                df['categorical_column'].value_counts()
                
                # Percentage distribution
                df['categorical_column'].value_counts(normalize=True) * 100
                
                # Crosstab analysis
                pd.crosstab(df['cat_col1'], df['cat_col2'])
                
                # Categorical column statistics by group
                df.groupby('categorical_column')['numerical_column'].describe()
                ```
                
                6. For data visualization:
                ```python
                # Bar chart
                import matplotlib.pyplot as plt
                df['categorical_column'].value_counts().plot(kind='bar')
                plt.title('Frequency of Categories')
                plt.xlabel('Category')
                plt.ylabel('Count')
                
                # Histogram
                plt.figure(figsize=(10, 6))
                df['numerical_column'].hist(bins=20)
                plt.title('Distribution of Values')
                
                # Scatter plot
                plt.figure(figsize=(10, 6))
                plt.scatter(df['x_column'], df['y_column'])
                plt.title('Relationship between X and Y')
                
                # Box plot
                plt.figure(figsize=(12, 6))
                df.boxplot(column=['num_col1', 'num_col2', 'num_col3'])
                plt.title('Distribution Comparison')
                
                # Group comparison
                import seaborn as sns
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='categorical_column', y='numerical_column', data=df)
                plt.title('Values by Category')
                ```
                
                Adapt the code examples above to answer the user's question using the appropriate column names from this dataset.
                """
                
                # For specific analytical questions like "top N", add specialized instructions
                if any(keyword in question.lower() for keyword in ["top", "highest", "max", "greatest", "lowest", "min"]):
                    prompt += """
                    
                    If the user is asking for the top/highest/lowest values or rankings:
                    1. Identify which column they want to rank by and which column contains the categories
                    2. Implement a solution with pandas operations - here's a template:
                    
                    ```python
                    # To find top N groups by some metric:
                    result = df.groupby('category_column')['value_column'].AGGREGATION_FUNCTION().sort_values(ascending=False).head(N)
                    
                    # Example: Top 5 countries by average income
                    top_countries = df.groupby('Country')['Income'].mean().sort_values(ascending=False).head(5)
                    
                    # Example: Lowest 3 products by total sales
                    bottom_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=True).head(3)
                    ```
                    
                    Replace 'category_column', 'value_column', and AGGREGATION_FUNCTION with the appropriate values from the dataset.
                    Show both the code and the results to answer the question directly.
                    """
            else:
                # Construct a more compact context from dataframe info
                num_cols = df_info.get('numerical_columns', [])[:3]
                cat_cols = df_info.get('categorical_columns', [])[:3]
                
                context = f"""
                Dataset: {df_info.get('rows', 'unknown')} rows, {df_info.get('columns', 'unknown')} columns.
                Key numerical columns: {', '.join(num_cols)}
                Key categorical columns: {', '.join(cat_cols)}
                
                Question: {question}
                """
                
                prompt = f"Answer about this dataset: {context}"
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Handle potential null response issues (like copyright detection)
            if not hasattr(response, 'text'):
                # Check if finish_reason is available
                if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason == 4:  # Copyright material detection
                        return """
I can't provide a specific answer as the data may contain copyrighted material. 

Try reformulating your question to be more general or analytical rather than asking for specific copyrighted content. For example:
- Ask for statistical analysis of the data
- Request grouping or aggregation of values
- Ask for correlations between columns
- Request insights or patterns in the data
"""
                    else:
                        return f"I couldn't generate a complete answer. Please try rephrasing your question to be more specific about the data analysis you need."
                else:
                    return "I couldn't generate a response. Please try rephrasing your question in a more analytical way."
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            error_message = str(e)
            if "copyrighted material" in error_message.lower() or "finish_reason" in error_message and "4" in error_message:
                return """
I can't provide a specific answer as the data may contain copyrighted material. 

Try reformulating your question to be more general or analytical rather than asking for specific copyrighted content. For example:
- Ask for statistical analysis of the data
- Request grouping or aggregation of values
- Ask for correlations between columns
- Request insights or patterns in the data
"""
            return f"I'm unable to answer this question. Please try rephrasing or check your Google API key in the .env file. Error: {str(e)}"
    
    def get_suggested_questions(self, data_profile):
        """Generate suggested follow-up questions based on the data profile"""
        if not self.model:
            return ["What insights can you find in this data?"]
        
        try:
            # Create prompt for suggested questions
            prompt = f"""
            Generate 5 interesting questions that could be asked about a dataset with the following characteristics:
            
            Dataset has {data_profile['rows']} rows and {data_profile['columns']} columns.
            
            Numerical columns: {', '.join(data_profile['numerical_columns'][:3])}
            Categorical columns: {', '.join(data_profile['categorical_columns'][:3])}
            
            Generate specific, data-focused questions that would lead to valuable insights.
            Make the questions concise and direct.
            Format each question as a separate line with no numbering or bullets.
            """
            
            # Get suggestions from the model
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Process the response
            questions = [q.strip() for q in response_text.split('\n') if q.strip()]
            return questions[:5]  # Return top 5 questions
            
        except Exception as e:
            logger.error(f"Error generating suggested questions: {e}")
            
            # Fallback to basic questions
            questions = []
            # Basic questions about the dataset
            questions.append("What insights can you find in this data?")
            
            # Questions about numerical columns
            if data_profile.get('numerical_columns'):
                num_cols = data_profile['numerical_columns'][:2]  # Limit to top 2
                for col in num_cols:
                    questions.append(f"What is the average {col}?")
                
                if len(num_cols) >= 2:
                    questions.append(f"Is there a relationship between {num_cols[0]} and {num_cols[1]}?")
            
            # Questions about categorical columns
            if data_profile.get('categorical_columns'):
                cat_cols = data_profile['categorical_columns'][:1]  # Limit to top 1
                for col in cat_cols:
                    questions.append(f"What are the most common {col} values?")
            
            # General analytical questions
            questions.append("What are the key patterns in this dataset?")
            
            return questions[:5]  # Return top 5 questions

    def get_fallback_insights(self, df, data_profile):
        """Generate basic insights when the AI model is not available"""
        insights = []
        
        # Basic dataset information
        insights.append(f"This dataset contains {data_profile['rows']} rows and {data_profile['columns']} columns.")
        
        # Missing values information
        if data_profile['missing_values_pct'] > 0:
            insights.append(f"The dataset has {data_profile['missing_values_pct']}% missing values.")
        else:
            insights.append("The dataset has no missing values.")
        
        # Numerical columns insights
        num_cols = data_profile.get('numerical_columns', [])[:3]
        for col in num_cols:
            try:
                mean_val = df[col].mean()
                max_val = df[col].max()
                min_val = df[col].min()
                insights.append(f"The column '{col}' has values ranging from {min_val:.2f} to {max_val:.2f}, with an average of {mean_val:.2f}.")
            except:
                pass
        
        # Categorical columns insights
        cat_cols = data_profile.get('categorical_columns', [])[:3]
        for col in cat_cols:
            try:
                top_category = df[col].value_counts().index[0]
                percentage = df[col].value_counts(normalize=True).iloc[0] * 100
                insights.append(f"The most common value in '{col}' is '{top_category}', representing {percentage:.1f}% of the data.")
            except:
                pass
        
        return insights 