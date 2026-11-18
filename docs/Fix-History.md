# Fix history

This log records user-visible fixes and the verification performed for each one.

## IA-001 — Reliable Gemini features and offline report fallback (2026-11-18)

**Problem.** The deployed app used the legacy `google-generativeai` package and an older model default. Its startup connectivity request could make AI availability appear broken, the key guidance referred only to `.env`, and report recommendations were not retained after an AI response.

**Resolution.** The app now uses Google's `google-genai` SDK with `gemini-3.5-flash-lite`, reads `GOOGLE_API_KEY` from Streamlit Secrets first (then the local environment), and creates no startup request. If a key is absent or the provider is unavailable, data profiling, cleaning, visualisations, local insights, chat guidance, and report recommendations remain available with their fallback status made clear. Generated report recommendations are now passed into the report builder.

**Verification.** Python compilation and offline unit tests cover the current GenAI client call contract, missing-key fallbacks, and report recommendations. The deployed app still needs `GOOGLE_API_KEY` in its Streamlit Community Cloud Secrets to make live Gemini calls; no secret is stored in this repository.
