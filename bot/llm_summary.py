# bot/llm_summary.py
import os
import requests
import time
from typing import List, Dict

# API Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

HEADERS = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

def generate_daily_summary(tips_data: List[Dict]) -> str:
    """
    Generate a summary of today's football tips using LLM.
    
    Args:
        tips_data: List of dicts with keys: home, away, league, best_tip, home_prob, draw_prob, away_prob
        
    Returns:
        LLM-generated summary message
    """
    if not TOGETHER_API_KEY:
        return "⚠️ LLM summary unavailable (missing API key)"
    
    if not tips_data:
        return "No high-confidence tips today."
    
    # Build prompt
    matches_text = "\n".join([
        f"- {tip['home']} vs {tip['away']} ({tip['league']}): {tip['best_tip']} "
        f"(Home: {tip['home_prob']:.0%}, Draw: {tip['draw_prob']:.0%}, Away: {tip['away_prob']:.0%})"
        for tip in tips_data
    ])
    
    prompt = f"""You are an expert football betting analyst. 
    Review the following high-confidence predictions for today's matches and provide a concise, professional summary.

    Key guidelines:
    - Focus on patterns (e.g., "strong home favorites dominate today")
    - Mention standout matches or leagues
    - Keep it under 150 words
    - Use confident, analytical language
    - Do NOT mention probabilities or percentages
    - No financial advice or stake suggestions
    - No exaggerated claims or guarantees

    Focus Areas:
    - Team form patterns and momentum
    - Historical context where relevant
    
    Today's predictions:
    {matches_text}

    Summary:"""
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.3,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        summary = result['choices'][0]['message']['content'].strip()
        
        # Clean up common LLM artifacts
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
            
        return summary
        
    except Exception as e:
        print(f"❌ LLM summary error: {e}")
        return "Unable to generate summary. Check API key and try again."

def create_summary_message(summary_text: str) -> str:
    """Format the LLM summary for Telegram with clean separators."""
    # Clean up any duplicate headers that might come from LLM
    clean_summary = summary_text.strip()
    if clean_summary.startswith("📊 DAILY TIPS SUMMARY"):
        clean_summary = clean_summary.replace("📊 DAILY TIPS SUMMARY", "").strip()
    
    return (
        "📊 TIPS SUMMARY\n"
        "──────────────────────────────\n"
        f"{clean_summary}\n\n"
        "💡 Bet responsibly || scoresignal"
    )