"""
Thai Translator

LLM-based English to Thai translation with financial terminology preservation.
"""

from __future__ import annotations

import os
import json
import hashlib
from typing import Dict, Any, Optional
import logging

from src.tools.llm import get_llm_client

logger = logging.getLogger(__name__)


class ThaiTranslator:
    """
    English to Thai translator specialized for financial reports

    Features:
    - LLM-based translation
    - Financial terminology glossary
    - Translation caching for consistency
    - Preserves numbers, dates, ticker symbols
    """

    def __init__(self, cache_dir: str = "cache/translations"):
        self.llm = get_llm_client()
        self.cache_dir = cache_dir
        self.glossary = self._load_financial_glossary()
        os.makedirs(cache_dir, exist_ok=True)

    def _load_financial_glossary(self) -> Dict[str, str]:
        """
        Financial terminology English → Thai mapping

        Ensures consistent translation of technical terms
        """
        return {
            # Options terminology
            "call option": "สิทธิซื้อ",
            "put option": "สิทธิขาย",
            "strike price": "ราคาใช้สิทธิ",
            "expiration date": "วันหมดอายุ",
            "expiration": "การหมดอายุ",
            "implied volatility": "ความผันผวนโดยนัย",
            "volatility": "ความผันผวน",
            "delta": "เดลต้า",
            "gamma": "แกมม่า",
            "theta": "ธีต้า",
            "vega": "เวก้า",
            "premium": "พรีเมี่ยม",
            "Greeks": "กรีกส์",

            # Strategies
            "bull call spread": "กลยุทธ์บูลคอลสเปรด",
            "bear put spread": "กลยุทธ์แบร์พุทสเปรด",
            "bull put spread": "กลยุทธ์บูลพุทสเปรด",
            "bear call spread": "กลยุทธ์แบร์คอลสเปรด",
            "iron condor": "กลยุทธ์ไอรอนคอนดอร์",
            "iron butterfly": "กลยุทธ์ไอรอนบัตเตอร์ฟลาย",
            "straddle": "กลยุทธ์สแตรดเดิล",
            "strangle": "กลยุทธ์สแตรงเกิล",
            "butterfly spread": "กลยุทธ์บัตเตอร์ฟลายสเปรด",
            "collar": "กลยุทธ์คอลลาร์",
            "long call": "ซื้อคอลออปชัน",
            "long put": "ซื้อพุทออปชัน",
            "short put": "ขายพุทออปชัน",

            # Market terms
            "bullish": "แนวโน้มขาขึ้น",
            "bearish": "แนวโน้มขาลง",
            "neutral": "แนวโน้มกลาง",
            "uptrend": "ขาขึ้น",
            "downtrend": "ขาลง",
            "liquidity": "สภาพคล่อง",
            "volume": "ปริมาณการซื้อขาย",
            "open interest": "ออเพนอินเทอเรสต์",

            # Risk terms
            "maximum risk": "ความเสี่ยงสูงสุด",
            "maximum profit": "กำไรสูงสุด",
            "break-even": "จุดคุ้มทุน",
            "probability of profit": "ความน่าจะเป็นที่จะกำไร",
            "risk-reward ratio": "อัตราส่วนความเสี่ยงต่อผลตอบแทน",

            # Technical indicators
            "moving average": "ค่าเฉลี่ยเคลื่อนที่",
            "RSI": "อาร์เอสไอ",
            "MACD": "เอ็มเอซีดี",
            "Bollinger Bands": "แถบโบลลิงเจอร์",
            "support": "แนวรับ",
            "resistance": "แนวต้าน",

            # Financial metrics
            "revenue": "รายได้",
            "earnings": "กำไร",
            "profit": "กำไร",
            "loss": "ขาดทุน",
            "growth": "การเติบโต",
            "margin": "อัตรากำไร",
            "debt": "หนี้สิน",
            "equity": "ส่วนของผู้ถือหุ้น",

            # General trading
            "trade": "การเทรด",
            "strategy": "กลยุทธ์",
            "position": "สถานะ",
            "portfolio": "พอร์ตโฟลิโอ",
            "market": "ตลาด",
            "price": "ราคา",
            "ticker": "สัญลักษณ์หุ้น",
            "stock": "หุ้น",
            "option": "ออปชัน",
            "contract": "สัญญา",
        }

    def translate_text(self, text: str, context: str = "") -> str:
        """
        Translate English text to Thai

        Args:
            text: English text to translate
            context: Context hint (e.g., "executive_summary", "risk_disclosure")

        Returns:
            Thai translation
        """
        if not text or len(text.strip()) < 3:
            return text

        # Check cache first
        cache_key = self._get_cache_key(text)
        cached = self._load_from_cache(cache_key)
        if cached:
            logger.debug(f"Using cached translation for: {text[:50]}...")
            return cached

        # Protect technical terms with placeholders
        protected_text, placeholders = self._protect_terms(text)

        # Translate using LLM
        try:
            thai_text = self._translate_with_llm(protected_text, context)

            # Restore technical terms
            for placeholder, thai_term in placeholders.items():
                thai_text = thai_text.replace(placeholder, thai_term)

            # Cache the translation
            self._save_to_cache(cache_key, thai_text)

            return thai_text

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text  # Fallback to English

    def _protect_terms(self, text: str) -> tuple[str, Dict[str, str]]:
        """
        Replace technical terms with placeholders

        Returns: (protected_text, placeholder_map)
        """
        protected = text
        placeholders = {}

        for idx, (en_term, th_term) in enumerate(self.glossary.items()):
            if en_term.lower() in text.lower():
                placeholder = f"__TERM_{idx}__"
                placeholders[placeholder] = th_term

                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(en_term), re.IGNORECASE)
                protected = pattern.sub(placeholder, protected)

        return protected, placeholders

    def _translate_with_llm(self, text: str, context: str) -> str:
        """
        Translate using LLM

        Args:
            text: Text with placeholders
            context: Context hint

        Returns:
            Thai translation
        """
        context_hint = f" (Context: {context})" if context else ""

        prompt = f"""
Translate the following financial text from English to Thai{context_hint}.

Requirements:
1. Translate naturally to Thai, maintaining professional financial tone
2. Keep numbers, dates, currency symbols ($, %), ticker symbols (e.g., AAPL, NVDA) unchanged
3. Preserve placeholders like __TERM_0__, __TERM_1__ exactly as-is (do not translate them)
4. Maintain paragraph structure and formatting
5. Use formal, professional Thai language suitable for investment reports

Text to translate:
{text}

Provide ONLY the Thai translation, no explanations or notes.
"""

        messages = [
            {
                "role": "system",
                "content": "You are a professional financial translator specializing in English to Thai translation for investment reports. You provide accurate, natural Thai translations while preserving technical terms and formatting."
            },
            {"role": "user", "content": prompt}
        ]

        thai_text = self.llm.chat(messages, temperature=0.2)
        return thai_text.strip()

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load translation from cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, translation: str):
        """Save translation to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(translation)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def translate_dict(self, data: Dict[str, Any], keys_to_translate: list[str]) -> Dict[str, Any]:
        """
        Translate specific keys in a dictionary

        Args:
            data: Dictionary to translate
            keys_to_translate: List of keys whose values should be translated

        Returns:
            Dictionary with translated values
        """
        translated = data.copy()

        for key in keys_to_translate:
            if key in translated and isinstance(translated[key], str):
                translated[key] = self.translate_text(translated[key], context=key)

        return translated
