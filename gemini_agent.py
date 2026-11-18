"""Gemini-backed explanations with useful, clearly-labelled local fallbacks."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import pandas as pd
import streamlit as st


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
MISSING_KEY_MESSAGE = (
    "AI assistance is unavailable because GOOGLE_API_KEY is not configured in "
    "Streamlit secrets. Dataset profiling, cleaning, charts, and offline report "
    "content still work."
)


def get_google_api_key() -> str | None:
    """Read a key without making local development depend on Streamlit secrets."""
    try:
        key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GOOGLE_API_KEY")


class GeminiAgent:
    """Generate prose from dataset context without executing generated code."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.client: Any | None = None
        self.initialized = False
        self.status_message = MISSING_KEY_MESSAGE

        key = api_key or get_google_api_key()
        if not key or key == "your_gemini_api_key_here":
            logger.info("Gemini is not configured; local analysis remains available.")
            return

        try:
            if client_factory is None:
                from google import genai

                client_factory = genai.Client
            self.client = client_factory(api_key=key)
            self.initialized = True
            self.status_message = "Gemini assistance is ready."
            logger.info("Configured Gemini model %s", model_name)
        except Exception:
            logger.exception("Could not initialize the Google GenAI client")
            self.status_message = (
                "AI assistance could not be initialized. Dataset profiling, cleaning, "
                "charts, and offline report content are still available."
            )

    def generate_text(self, prompt: str) -> str | None:
        """Return provider text, retaining a safe status for UI fallback on failure."""
        if not self.client:
            return None
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if text and text.strip():
                return text.strip()
            self.status_message = "Gemini returned no text. Try again or use the offline analysis."
        except Exception:
            logger.exception("Gemini generation failed")
            self.status_message = (
                "Gemini did not complete this request. Check the configured key and "
                "provider availability, or use the offline analysis."
            )
        return None

    def generate_data_summary(self, data_profile: dict[str, Any], df_sample: pd.DataFrame) -> str:
        prompt = f"""
You are a careful data analyst. Summarize this dataset in three or four concise sentences.
Use only the supplied facts. Do not claim to have executed code or inspected rows beyond
the sample.

Rows: {data_profile['rows']}
Columns: {data_profile['columns']}
Missing values: {data_profile['missing_values_pct']}%
Numerical columns: {', '.join(data_profile.get('numerical_columns', [])[:5]) or 'none'}
Categorical columns: {', '.join(data_profile.get('categorical_columns', [])[:5]) or 'none'}
Sample rows:
{df_sample.head(3).to_string()}
"""
        return self.generate_text(prompt) or self._fallback_summary(data_profile)

    def generate_insights(
        self,
        df: pd.DataFrame,
        visualizations: list[Any],
        data_profile: dict[str, Any],
    ) -> list[str]:
        numerical = data_profile.get("numerical_columns", [])[:5]
        categorical = data_profile.get("categorical_columns", [])[:5]
        number_stats = []
        for column in numerical:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if not series.empty:
                number_stats.append(
                    f"{column}: min={series.min():.2f}, max={series.max():.2f}, "
                    f"mean={series.mean():.2f}, median={series.median():.2f}"
                )
        category_stats = []
        for column in categorical:
            values = df[column].dropna().value_counts().head(2)
            if not values.empty:
                category_stats.append(
                    f"{column}: " + ", ".join(f"{value} ({count})" for value, count in values.items())
                )

        prompt = f"""
Give four to five concise, data-grounded insights. Use only these facts and do not invent
causes or business outcomes. Put one insight on each line.

Dataset: {data_profile['rows']} rows, {data_profile['columns']} columns; missing values: {data_profile['missing_values_pct']}%.
Numerical statistics:
{chr(10).join(number_stats) or 'No usable numerical statistics.'}
Categorical counts:
{chr(10).join(category_stats) or 'No usable categorical counts.'}
Sample:
{df.head(5).to_string()}
"""
        text = self.generate_text(prompt)
        if text:
            insights = [line.strip(" -•0123456789.") for line in text.splitlines() if line.strip()]
            if insights:
                return insights[:5]
        return self.get_fallback_insights(df, data_profile)

    def answer_question(self, question: str, df_info: dict[str, Any]) -> str:
        """Answer with supplied context only; Pandas snippets are suggestions."""
        prompt = df_info.get("full_prompt")
        if not prompt:
            prompt = f"""
Dataset: {df_info.get('rows', 'unknown')} rows, {df_info.get('columns', 'unknown')} columns.
Numerical columns: {', '.join(df_info.get('numerical_columns', [])[:5]) or 'none'}
Categorical columns: {', '.join(df_info.get('categorical_columns', [])[:5]) or 'none'}
Question: {question}
"""
        prompt += """
Answer only from the supplied dataset context. Do not say that code was executed. If you
include Pandas code, label it as a suggestion and use only listed column names. State when
the context is insufficient to answer the question.
"""
        text = self.generate_text(prompt)
        if text:
            return text
        return (
            f"{self.status_message}\n\n"
            "I cannot provide an AI-generated answer right now. You can still inspect the "
            "computed profile and charts, or configure a Google API key and try again."
        )

    def generate_recommendations(
        self,
        data_profile: dict[str, Any],
        insights: list[str],
    ) -> list[str]:
        prompt = f"""
Give three concise, cautious next-analysis recommendations based only on this dataset profile.
Do not invent business context. Put one recommendation on each line.

Rows: {data_profile['rows']}; columns: {data_profile['columns']}; missing values: {data_profile['missing_values_pct']}%.
Numerical columns: {', '.join(data_profile.get('numerical_columns', [])[:5]) or 'none'}
Categorical columns: {', '.join(data_profile.get('categorical_columns', [])[:5]) or 'none'}
Existing insights: {' '.join(insights[:3]) or 'none'}
"""
        text = self.generate_text(prompt)
        if text:
            recommendations = [line.strip(" -•0123456789.") for line in text.splitlines() if line.strip()]
            if recommendations:
                return recommendations[:5]

        recommendations = ["Review the dataset profile and validate column definitions before drawing conclusions."]
        if data_profile.get("missing_values_pct", 0) > 0:
            recommendations.append("Decide how to handle missing values and document the chosen rule.")
        if len(data_profile.get("numerical_columns", [])) >= 2:
            recommendations.append("Compare numerical columns with a scatter plot or correlation check before inferring relationships.")
        if data_profile.get("categorical_columns"):
            recommendations.append("Review category counts for rare or imbalanced values before aggregating results.")
        return recommendations[:5]

    def get_suggested_questions(self, data_profile: dict[str, Any]) -> list[str]:
        numerical = data_profile.get("numerical_columns", [])[:2]
        categorical = data_profile.get("categorical_columns", [])[:1]
        prompt = f"""
Suggest up to five short, specific questions for a dataset with numerical columns
{', '.join(numerical) or 'none'} and categorical columns {', '.join(categorical) or 'none'}.
Put one question per line.
"""
        text = self.generate_text(prompt)
        if text:
            questions = [line.strip(" -•0123456789.") for line in text.splitlines() if line.strip()]
            if questions:
                return questions[:5]
        questions = ["What are the key patterns in this dataset?"]
        questions.extend(f"What is the average {column}?" for column in numerical)
        if len(numerical) >= 2:
            questions.append(f"Is there a relationship between {numerical[0]} and {numerical[1]}?")
        if categorical:
            questions.append(f"What are the most common values of {categorical[0]}?")
        return questions[:5]

    @staticmethod
    def _fallback_summary(data_profile: dict[str, Any]) -> str:
        numerical = ", ".join(data_profile.get("numerical_columns", [])[:5]) or "no numerical columns"
        categorical = ", ".join(data_profile.get("categorical_columns", [])[:5]) or "no categorical columns"
        return (
            f"This dataset contains {data_profile['rows']} rows and {data_profile['columns']} columns. "
            f"It has {data_profile['missing_values_pct']}% missing values. "
            f"The available numerical columns are {numerical}; categorical columns are {categorical}. "
            "This summary was generated locally from the dataset profile."
        )

    @staticmethod
    def get_fallback_insights(df: pd.DataFrame, data_profile: dict[str, Any]) -> list[str]:
        insights = [
            f"Local profile: {data_profile['rows']} rows and {data_profile['columns']} columns.",
            (
                f"Missing values account for {data_profile['missing_values_pct']}% of cells."
                if data_profile.get("missing_values_pct", 0) > 0
                else "The local profile found no missing values."
            ),
        ]
        for column in data_profile.get("numerical_columns", [])[:3]:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if not series.empty:
                insights.append(
                    f"{column} ranges from {series.min():.2f} to {series.max():.2f}, with a mean of {series.mean():.2f}."
                )
        for column in data_profile.get("categorical_columns", [])[:2]:
            values = df[column].dropna().value_counts()
            if not values.empty:
                top_value, top_count = values.index[0], values.iloc[0]
                insights.append(
                    f"The most common {column} value is {top_value!r}, appearing {top_count} times."
                )
        return insights
