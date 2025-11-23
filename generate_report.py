import argparse
import base64
import io
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import markdown
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from src.tools.llm import get_llm_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REPORT_FILES: Dict[str, str] = {
    "fundamental_report": "fundamental_report.json",
    "technical_report": "technical_report.json",
    "trade_decision": "trade_decision.json",
    "trade_thesis": "trade_thesis.json",
    "volatility_report": "volatility_report.json",
    "sentiment_report": "sentiment_report.json",
    "risk_assessment": "risk_assessment.json",
}

JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
SPECIAL_ACRONYMS = {"iv", "rsi", "macd", "aws", "obv", "fcf", "ema", "sma"}

# Strategy explanations for beginners
STRATEGY_EXPLANATIONS = {
    "Bull Call Spread": {
        "explanation": "A Bull Call Spread is a moderate-risk strategy where you buy a call option at a lower strike price and sell another call at a higher strike price. This reduces your upfront cost compared to buying a call alone, but also caps your maximum profit. It's ideal when you expect the stock to rise moderately.",
        "pros_cons": "✓ Limited risk (only the premium paid) ✓ Lower cost than buying calls alone ✓ Profits from moderate upward moves · ✗ Capped profit potential ✗ Loses value if stock doesn't rise"
    },
    "Bear Put Spread": {
        "explanation": "A Bear Put Spread involves buying a put option at a higher strike price and selling another put at a lower strike price. This strategy profits when the stock declines, with limited risk and limited reward. It's cheaper than buying puts outright.",
        "pros_cons": "✓ Limited risk (only the premium paid) ✓ Lower cost than buying puts alone ✓ Profits from moderate downward moves · ✗ Capped profit potential ✗ Loses value if stock doesn't fall"
    },
    "Iron Condor": {
        "explanation": "An Iron Condor combines a Bear Call Spread and a Bull Put Spread. You collect premium upfront and profit if the stock stays within a specific range. This is a neutral strategy that benefits from low volatility and time decay.",
        "pros_cons": "✓ Collects premium upfront ✓ Profits from range-bound markets ✓ High probability of profit · ✗ Limited profit potential ✗ Requires active management ✗ Larger losses if stock moves significantly"
    },
    "Cash-Secured Put": {
        "explanation": "Selling a Cash-Secured Put means you sell a put option while holding enough cash to buy the stock if assigned. You collect premium immediately. If the stock stays above the strike price, you keep the premium. If it falls below, you buy the stock at the strike price.",
        "pros_cons": "✓ Generates income from premiums ✓ Lets you buy stocks at a discount ✓ Lower risk than owning stock outright · ✗ Requires significant capital ✗ Unlimited loss potential if stock crashes"
    },
    "Covered Call": {
        "explanation": "A Covered Call involves owning 100 shares of stock and selling a call option against them. You collect premium income while potentially selling your shares at a higher price. This strategy generates income from stocks you already own.",
        "pros_cons": "✓ Generates regular income ✓ Reduces cost basis of stock ✓ Works well in flat/slightly bullish markets · ✗ Caps upside profit ✗ You may have to sell your shares ✗ Still exposed to downside risk"
    },
    "Long Call": {
        "explanation": "Buying a Long Call gives you the right (but not obligation) to buy stock at a specific price before expiration. This is a bullish strategy with limited risk (the premium paid) and unlimited profit potential. It's like placing a leveraged bet on the stock rising.",
        "pros_cons": "✓ Unlimited profit potential ✓ Limited risk (only premium paid) ✓ Requires less capital than buying stock · ✗ Loses value over time (theta decay) ✗ Requires stock to move significantly to profit"
    },
    "Long Put": {
        "explanation": "Buying a Long Put gives you the right to sell stock at a specific price. This is a bearish strategy or can be used to protect stock you own. Your risk is limited to the premium paid, while profit potential is substantial if the stock falls.",
        "pros_cons": "✓ Substantial profit potential ✓ Limited risk (only premium paid) ✓ Can hedge stock positions · ✗ Loses value over time (theta decay) ✗ Requires stock to move significantly to profit"
    },
    "Straddle": {
        "explanation": "A Straddle involves buying both a call and a put at the same strike price. You profit from large moves in either direction. This is useful before earnings or major announcements when you expect big volatility but don't know the direction.",
        "pros_cons": "✓ Profits from big moves in any direction ✓ Unlimited profit potential · ✗ Expensive (buying two options) ✗ Loses money if stock doesn't move enough ✗ Heavy time decay"
    },
    "Strangle": {
        "explanation": "A Strangle is like a Straddle but uses different strike prices (out-of-the-money). It's cheaper than a Straddle but requires larger moves to profit. Good for expecting significant volatility at lower cost.",
        "pros_cons": "✓ Cheaper than Straddle ✓ Profits from big moves either direction · ✗ Requires very large moves to profit ✗ Loses money from time decay ✗ Can lose entire premium"
    },
    "Butterfly Spread": {
        "explanation": "A Butterfly Spread uses three strike prices to create a position that profits most when the stock stays near the middle strike. It's a low-cost, limited-risk way to bet on low volatility and minimal price movement.",
        "pros_cons": "✓ Very low cost ✓ Limited risk ✓ High reward-to-risk ratio at middle strike · ✗ Very small profit zone ✗ Low probability of maximum profit"
    },
    "Bear Call Spread": {
        "explanation": "A Bear Call Spread involves selling a call option at a lower strike price and buying another call at a higher strike price. You collect a premium upfront (net credit). This strategy profits when the stock price stays below the lower strike price.",
        "pros_cons": "✓ Income generation (credit received) ✓ Defined risk (spread width minus credit) · ✗ Capped profit (limited to credit received) ✗ Losses occur if stock rallies significantly"
    },
    "Bull Put Spread": {
        "explanation": "A Bull Put Spread involves selling a put option at a higher strike price and buying another put at a lower strike price. You collect a premium upfront. This strategy profits when the stock stays above the higher strike price.",
        "pros_cons": "✓ Income generation (credit received) ✓ Defined risk · ✗ Capped profit ✗ Losses occur if stock drops significantly"
    }
}

# Thai strategy explanations (natural mixed Thai-English style)
STRATEGY_EXPLANATIONS_TH = {
    "Bull Call Spread": {
        "explanation": "Bull Call Spread คือซื้อ Call ที่ Strike ต่ำกว่า และขาย Call อีกตัวที่ Strike สูงกว่า ทำให้ต้นทุนถูกกว่าซื้อ Call เพียงอย่างเดียว แต่กำไรสูงสุดจะถูก cap ไว้ เหมาะกับตอนที่คาดว่าราคาหุ้นจะขึ้นแบบ moderate",
        "pros_cons": "✓ ความเสี่ยงจำกัด (เสียแค่ Premium ที่จ่าย)\n✓ ต้นทุนต่ำกว่าซื้อ Call แบบเดี่ยว\n✓ ได้กำไรจากราคาหุ้นที่ขึ้นปานกลาง\n✗ กำไรสูงสุดถูก cap ไว้\n✗ ขาดทุนถ้าหุ้นไม่ขึ้น"
    },
    "Bear Put Spread": {
        "explanation": "Bear Put Spread คือซื้อ Put ที่ Strike สูงกว่า และขาย Put อีกตัวที่ Strike ต่ำกว่า ทำกำไรเมื่อหุ้นลง มีความเสี่ยงและผลตอบแทนที่จำกัด ต้นทุนถูกกว่าซื้อ Put เพียงอย่างเดียว",
        "pros_cons": "✓ ความเสี่ยงจำกัด (เสียแค่ Premium)\n✓ ต้นทุนต่ำกว่าซื้อ Put แบบเดี่ยว\n✓ ทำกำไรเมื่อหุ้นลงปานกลาง\n✗ กำไรสูงสุดถูก cap ไว้\n✗ ขาดทุนถ้าหุ้นไม่ลง"
    },
    "Iron Condor": {
        "explanation": "Iron Condor รวม Bear Call Spread และ Bull Put Spread เข้าด้วยกัน ได้ Premium ล่วงหน้า ทำกำไรถ้าหุ้นอยู่ในช่วงที่กำหนด เป็นกลยุทธ์แบบ neutral ที่ได้ประโยชน์จาก Volatility ต่ำและ Time Decay",
        "pros_cons": "✓ ได้ Premium ล่วงหน้า\n✓ ทำกำไรเมื่อตลาดไม่มีทิศทางชัด\n✓ Probability of profit สูง\n✗ กำไรจำกัด\n✗ ต้องดูแลใกล้ชิด\n✗ ขาดทุนมากถ้าหุ้นเคลื่อนไหวแรง"
    },
    "Cash-Secured Put": {
        "explanation": "ขาย Cash-Secured Put คือขาย Put โดยมีเงินสดพร้อมซื้อหุ้นถ้าถูก assign ได้ Premium ทันที ถ้าหุ้นอยู่เหนือ Strike ได้กำไรจาก Premium ถ้าต่ำกว่าต้องซื้อหุ้นที่ Strike",
        "pros_cons": "✓ สร้างรายได้จาก Premium\n✓ ซื้อหุ้นในราคาที่ต่ำกว่าตลาด\n✓ เสี่ยงต่ำกว่าถือหุ้นตรง\n✗ ต้องใช้เงินทุนมาก\n✗ เสี่ยงไม่จำกัดถ้าหุ้นร่วง"
    },
    "Covered Call": {
        "explanation": "Covered Call คือถือหุ้น 100 หุ้น และขาย Call กับหุ้นนั้น ได้รายได้จาก Premium พร้อมโอกาสขายหุ้นในราคาสูงขึ้น กลยุทธ์นี้สร้างรายได้จากหุ้นที่ถืออยู่",
        "pros_cons": "✓ สร้างรายได้สม่ำเสมอ\n✓ ลดต้นทุนการถือหุ้น\n✓ เหมาะกับตลาดที่ sideways หรือขึ้นเล็กน้อย\n✗ จำกัดกำไรขาขึ้น\n✗ อาจต้องขายหุ้นถ้าราคาพุ่ง\n✗ ยังเสี่ยงขาลง"
    },
    "Long Call": {
        "explanation": "ซื้อ Long Call ให้สิทธิ์ซื้อหุ้นที่ Strike ที่กำหนดก่อนหมดอายุ เป็นกลยุทธ์แบบ bullish ที่เสี่ยงจำกัด (เสียแค่ Premium) แต่กำไรไม่จำกัด เหมือน leverage bet ว่าหุ้นจะขึ้น",
        "pros_cons": "✓ กำไรไม่จำกัด\n✓ ความเสี่ยงจำกัด (เสียแค่ Premium)\n✓ ใช้เงินทุนน้อยกว่าซื้อหุ้นตรง\n✗ Theta Decay กัดกินมูลค่า\n✗ ต้องการให้หุ้นเคลื่อนไหวมาก"
    },
    "Long Put": {
        "explanation": "ซื้อ Long Put ให้สิทธิ์ขายหุ้นที่ Strike ที่กำหนด เป็นกลยุทธ์แบบ bearish หรือใช้ hedge หุ้นที่ถือ เสี่ยงจำกัดที่ Premium กำไรมีศักยภาพสูงถ้าหุ้นลง",
        "pros_cons": "✓ กำไรมีศักยภาพสูง\n✓ ความเสี่ยงจำกัด (เสียแค่ Premium)\n✓ ป้องกันพอร์ตหุ้น\n✗ Theta Decay กัดกินมูลค่า\n✗ ต้องการให้หุ้นเคลื่อนไหวมาก"
    },
    "Straddle": {
        "explanation": "Straddle คือซื้อทั้ง Call และ Put ที่ Strike เดียวกัน ทำกำไรจากการเคลื่อนไหวครั้งใหญ่ไปทิศทางไหนก็ได้ เหมาะก่อน earnings หรือข่าวใหญ่ เมื่อคาดว่า Volatility จะสูงแต่ไม่รู้ทิศทาง",
        "pros_cons": "✓ กำไรจากการเคลื่อนไหวแรงทุกทิศทาง\n✓ กำไรไม่จำกัด\n✗ แพง (ซื้อสอง Option)\n✗ ขาดทุนถ้าหุ้นไม่เคลื่อนไหว\n✗ Time Decay สูง"
    },
    "Strangle": {
        "explanation": "Strangle คล้าย Straddle แต่ใช้ Strike ต่างกัน (OTM) ถูกกว่า Straddle แต่ต้องการการเคลื่อนไหวมากกว่าถึงจะทำกำไร เหมาะเมื่อคาดว่า Volatility จะสูงในราคาที่ต่ำกว่า",
        "pros_cons": "✓ ถูกกว่า Straddle\n✓ กำไรจากการเคลื่อนไหวแรงทุกทิศทาง\n✗ ต้องการการเคลื่อนไหวมากมาก\n✗ Time Decay กัดกิน\n✗ อาจเสีย Premium ทั้งหมด"
    },
    "Butterfly Spread": {
        "explanation": "Butterfly Spread ใช้สาม Strike สร้าง position ที่ทำกำไรสูงสุดเมื่อหุ้นอยู่ใกล้ Strike กลาง ต้นทุนต่ำ เสี่ยงจำกัด เหมาะกับการ bet ว่าหุ้นจะมี Volatility ต่ำและเคลื่อนไหวน้อย",
        "pros_cons": "✓ ต้นทุนต่ำมาก\n✓ ความเสี่ยงจำกัด\n✓ Risk/reward ratio สูงที่ Strike กลาง\n✗ Profit zone แคบมาก\n✗ Probability of max profit ต่ำ"
    },
    "Bear Call Spread": {
        "explanation": "Bear Call Spread คือขาย Call ที่ Strike ต่ำกว่า และซื้อ Call อีกตัวที่ Strike สูงกว่า ได้ Premium ล่วงหน้า (Net Credit) ทำกำไรเมื่อหุ้นอยู่ต่ำกว่า Strike ที่ขาย เป็นกลยุทธ์แบบ bearish ที่เก็บเงิน Premium",
        "pros_cons": "✓ สร้างรายได้จาก Premium\n✓ ความเสี่ยงจำกัด (ความกว้างของ Spread ลบ Credit)\n✗ กำไรถูก cap ที่ Premium ที่ได้\n✗ ขาดทุนถ้าหุ้นพุ่งขึ้น"
    },
    "Bull Put Spread": {
        "explanation": "Bull Put Spread คือขาย Put ที่ Strike สูงกว่า และซื้อ Put อีกตัวที่ Strike ต่ำกว่า ได้ Premium ล่วงหน้า ทำกำไรเมื่อหุ้นอยู่เหนือ Strike ที่ขาย เป็นกลยุทธ์แบบ bullish ที่เก็บเงิน Premium",
        "pros_cons": "✓ สร้างรายได้จาก Premium\n✓ ความเสี่ยงจำกัด\n✗ กำไรถูก cap\n✗ ขาดทุนถ้าหุ้นร่วง"
    }
}


def translate_to_thai_simple(text: str, context: str = "general") -> str:
    """
    Simple, focused translation for short texts
    """
    if not text or not text.strip() or len(text) < 10:
        return text

    llm = get_llm_client()

    prompt = f"""Translate this {context} to natural Thai-English mixed style:

"{text}"

Rules:
- Keep technical terms in English: Delta, Gamma, Theta, Vega, IV, RSI, MACD, P/E, EPS, Strike, Premium, Call, Put
- Use natural conversational Thai for explanations
- NO markdown (###, |), just clean text
- Output ONLY the Thai translation, nothing else"""

    try:
        response = llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
        result = response.strip()

        # Clean up
        result = result.replace('###', '').replace('##', '').replace(' | ', ' ')
        result = result.replace('"', '').replace("'", "")

        return result if result else text
    except:
        return text


def translate_to_thai(text: str, context_type: str = "general") -> str:
    """
    Translate text to Thai by splitting into chunks for better quality
    """
    if not text or not text.strip() or len(text) < 20:
        return text

    # Remove markdown artifacts BEFORE translation
    import re
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Remove headers
    text = re.sub(r'\|[^\n]+\|', '', text)  # Remove table rows
    text = text.replace('---', '').replace('***', '')  # Remove separators

    # Split into paragraphs for better translation
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    if not paragraphs:
        return text

    # If text is short enough, translate in one go
    if len(text) < 500:
        return _translate_chunk(text, context_type)

    # Otherwise, translate paragraph by paragraph
    translated_paragraphs = []
    for para in paragraphs:
        if len(para) < 20:  # Skip very short paragraphs
            translated_paragraphs.append(para)
            continue

        translated = _translate_chunk(para, context_type)
        if translated:
            translated_paragraphs.append(translated)

    return '\n\n'.join(translated_paragraphs)


def _translate_chunk(text: str, context: str) -> str:
    """
    Translate a single chunk of text to Thai
    """
    if not text or not text.strip():
        return text

    llm = get_llm_client()

    prompt = f"""แปลข้อความนี้เป็นภาษาไทย แบบธรรมชาติ ผสม Thai-English:

{text}

กฎ:
1. ใช้ภาษาไทยที่เป็นธรรมชาติ ไม่เหมือนหุ่นยนต์
2. เก็บศัพท์เทคนิคเป็นภาษาอังกฤษ: Delta, Gamma, IV, RSI, MACD, P/E, EPS, Strike, Premium, Call, Put, FOMC, Fed
3. ห้ามใส่ markdown (###, |, ---)
4. ห้ามใส่ label อย่าง "English text:" หรือ "Thai translation:"
5. แปลให้หมดทุกส่วน อย่าทิ้งภาษาอังกฤษไว้
6. ตอบแค่ข้อความภาษาไทยเท่านั้น

Output เป็นภาษาไทยเท่านั้น:"""

    try:
        response = llm.chat([{"role": "user", "content": prompt}], temperature=0.2)
        result = response.strip()

        # Clean up any artifacts
        result = result.replace('###', '').replace('##', '').replace('#', '')
        result = result.replace(' | ', ' ').replace('|', '')
        result = result.replace('---', '').replace('***', '')
        result = result.replace('English text:', '').replace('Thai translation:', '')
        result = result.replace('"', '').replace("'", "")

        # Remove lines that are just markdown or debug
        lines = [l for l in result.split('\n') if l.strip() and not l.strip().startswith(('---', '***', '###'))]
        result = '\n'.join(lines)

        return result.strip() if result.strip() else text

    except Exception as e:
        print(f"Translation error for chunk: {e}")
        return text


def load_all_reports(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for key, filename in REPORT_FILES.items():
        file_path = os.path.join(path, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Missing required report file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as handle:
            data[key] = json.load(handle)
    return data


def extract_json_blocks(text: Optional[str]) -> List[Any]:
    if not isinstance(text, str):
        return []
    blocks: List[Any] = []
    for match in JSON_BLOCK_RE.finditer(text):
        try:
            blocks.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return blocks


def strip_code_blocks(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    return JSON_BLOCK_RE.sub("", text).strip()


def render_markdown_html(text: Optional[str]) -> str:
    cleaned = strip_code_blocks(text)
    if not cleaned:
        return ""
    return markdown.markdown(
        cleaned,
        extensions=["extra", "sane_lists", "tables", "fenced_code"],
        output_format="html5",
    )


def humanize_phrase(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    words: List[str] = []
    for word in text.split(" "):
        lower = word.lower()
        if lower in SPECIAL_ACRONYMS:
            words.append(lower.upper())
        elif lower.isupper():
            words.append(lower)
        else:
            words.append(lower.capitalize())
    return " ".join(words)


def get_nested(data: Any, path: Sequence[Any], default: Optional[Any] = None) -> Optional[Any]:
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return default
    return current


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def format_timestamp(value: Optional[Any]) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = parse_iso_datetime(value)
    else:
        dt = None
    if not dt:
        return value if isinstance(value, str) and value else "N/A"
    fmt = "%B %d, %Y %H:%M %Z" if dt.tzinfo else "%B %d, %Y %H:%M"
    return dt.strftime(fmt)


def format_date(value: Optional[Any]) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = parse_iso_datetime(value)
    else:
        dt = None
    if not dt:
        return value if isinstance(value, str) and value else "N/A"
    return dt.strftime("%B %d, %Y")


def _calculate_dte(expiration: Optional[Any], report_generated_at: Optional[Any]) -> Optional[int]:
    exp_dt: Optional[datetime] = None
    if isinstance(expiration, str):
        exp_dt = parse_iso_datetime(expiration)
    elif isinstance(expiration, datetime):
        exp_dt = expiration
    base_dt = parse_iso_datetime(report_generated_at) if isinstance(report_generated_at, str) else None
    if base_dt is None:
        base_dt = datetime.now()
    if exp_dt is None:
        return None
    delta = exp_dt.date() - base_dt.date()
    return max(delta.days, 0)


def format_currency(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"${number:,.{decimals}f}"


def format_large_currency(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    abs_value = abs(number)
    for divisor, suffix in ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs_value >= divisor:
            return f"${number / divisor:,.2f}{suffix}"
    if abs_value >= 1000:
        return f"${number:,.0f}"
    return f"${number:,.2f}"


def format_number(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}"
    fmt = f"{{:,.{decimals}f}}"
    return fmt.format(number)


def format_percentage(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) <= 1:
        number *= 100
    return f"{number:.{decimals}f}%"


def clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def flatten_details(details: Dict[str, Any], skip_keys: Optional[Iterable[str]] = None, prefix: str = "") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    skipped = set(skip_keys or [])
    for key, value in details.items():
        if key in skipped:
            continue
        label = humanize_phrase(key) if not prefix else f"{prefix} – {humanize_phrase(key)}"
        if isinstance(value, dict):
            rows.extend(flatten_details(value, skip_keys=skip_keys, prefix=label))
        elif isinstance(value, list):
            joined = ", ".join(format_number(item) if isinstance(item, (int, float)) else str(item) for item in value)
            rows.append({"label": label, "value": joined})
        else:
            display = format_number(value) if isinstance(value, (int, float)) else str(value)
            rows.append({"label": label, "value": display})
    return rows


def infer_report_datetime(reports: Dict[str, Any], timestamp: Optional[str] = None) -> datetime:
    """
    Infer report datetime with preference order:
    1. timestamp parameter (from folder name) - most reliable
    2. generated_at from JSON files
    3. datetime.now() as fallback

    FIX: Expert review - prefer timestamp from folder name for consistency
    """
    # FIX: First try to parse timestamp from folder name (e.g., "20251123_114451")
    if timestamp:
        try:
            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            return dt
        except (ValueError, TypeError):
            pass  # If parsing fails, fall through to JSON timestamps

    # Fallback to generated_at from JSON files
    for key in ("trade_decision", "trade_thesis", "fundamental_report", "technical_report"):
        generated = reports.get(key, {}).get("generated_at")
        dt = parse_iso_datetime(generated)
        if dt:
            return dt

    # Last resort: use current time
    return datetime.now()


def extract_trade_summary(trade_decision: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "max_risk": None,
        "max_reward": None,
        "justification": None,
        "details": {},
        "raw_notes": None,
        "conviction": None,
        "direction": None,
        "holding_period": None,
        "cost_per_contract": None,
        "computed_dte": None,
    }
    notes = trade_decision.get("notes")
    proposal: Dict[str, Any] = {}
    if isinstance(notes, dict):
        proposal = notes.get("trade_proposal") or notes
    elif isinstance(notes, str):
        for block in extract_json_blocks(notes):
            if isinstance(block, dict):
                if "trade_proposal" in block and isinstance(block["trade_proposal"], dict):
                    proposal = block["trade_proposal"]
                else:
                    proposal = block
                break
        summary["raw_notes"] = notes
    if proposal:
        summary["details"] = proposal
        summary["max_risk"] = (
            proposal.get("max_risk")
            or proposal.get("max_cost")
            or proposal.get("max_cost_per_share")
            or proposal.get("max_loss")
        )
        summary["max_reward"] = proposal.get("max_reward") or proposal.get("max_profit")
        summary["justification"] = proposal.get("justification") or proposal.get("rationale")
        summary["conviction"] = proposal.get("conviction")
        summary["direction"] = proposal.get("thesis_direction") or proposal.get("direction")
        summary["holding_period"] = proposal.get("days_to_expiration") or proposal.get("holding_period")
        summary["cost_per_contract"] = proposal.get("estimated_cost_per_contract")
        if summary["max_reward"] is None and (proposal.get("net_credit_debit") or "").lower().startswith("credit"):
            summary["max_reward"] = proposal.get("net_premium")
    elif isinstance(notes, str):
        stripped = strip_code_blocks(notes)
        if stripped:
            summary["justification"] = stripped
    summary["justification"] = summary["justification"] or trade_decision.get("strategy_rationale")
    expiration = get_nested(trade_decision, ["trade_legs", 0, "expiration_date"])
    summary["computed_dte"] = _calculate_dte(expiration, trade_decision.get("generated_at"))
    return summary


def build_derived_fields(reports: Dict[str, Any]) -> Dict[str, Any]:
    latest_close = get_nested(reports, ["technical_report", "indicators", "latest_close"])
    price_date = get_nested(reports, ["technical_report", "indicators", "price_date"])
    current_price = get_nested(reports, ["trade_thesis", "current_price"])
    return {
        "latest_close": latest_close,
        "price_date": price_date,
        "price_date_human": format_date(price_date),
        "current_price": current_price if current_price is not None else latest_close,
        "rsi_value": get_nested(reports, ["technical_report", "indicators", "RSI", "value"]),
        "rsi_regime": get_nested(reports, ["technical_report", "indicators", "RSI", "regime"]),
        "macd_histogram": get_nested(reports, ["technical_report", "indicators", "MACD_Signal", "histogram"]),
        "macd_crossover": get_nested(reports, ["technical_report", "indicators", "MACD_Signal", "crossover"]),
        "supertrend_level": get_nested(reports, ["technical_report", "indicators", "Supertrend_Signal", "level"]),
        "supertrend_trend": get_nested(reports, ["technical_report", "indicators", "Supertrend_Signal", "trend"]),
        "obv_value": get_nested(reports, ["technical_report", "indicators", "OBV_Trend", "value"]),
        "obv_trend": get_nested(reports, ["technical_report", "indicators", "OBV_Trend", "trend"]),
        "technical_bias": get_nested(reports, ["technical_report", "llm_report", "technical_bias"]),
        "iv_rank": get_nested(reports, ["volatility_report", "iv_rank"]),
        "volatility_forecast": get_nested(reports, ["volatility_report", "volatility_forecast"]),
        "sentiment_score": get_nested(reports, ["sentiment_report", "overall_sentiment_score"]),
        "sentiment_summary": get_nested(reports, ["sentiment_report", "overall_summary"]),
        "market_cap": get_nested(reports, ["fundamental_report", "business_overview", "marketCap"]),
    }


def build_market_snapshot_rows(derived: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = [
        {"label": "Price", "value": format_currency(derived.get("current_price"))},
        {"label": "Price Date", "value": derived.get("price_date_human") or "N/A"},
        {"label": "IV Rank", "value": format_number(derived.get("iv_rank"), 1) if derived.get("iv_rank") is not None else "N/A"},
        {"label": "Technical Bias", "value": humanize_phrase(derived.get("technical_bias")) or "Neutral"},
        {"label": "Sentiment", "value": derived.get("sentiment_summary") or "Neutral"},
    ]
    return rows


def build_trade_section(reports: Dict[str, Any], derived: Dict[str, Any], summary: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    decision = reports.get("trade_decision", {}) or {}
    summary_details = summary.get("details", {}) or {}
    direction = summary.get("direction") or summary_details.get("direction") or decision.get("action")
    strategy = summary_details.get("strategy") or decision.get("strategy_name")
    conviction = (
        decision.get("conviction_level")
        or summary.get("conviction")
        or summary_details.get("conviction")
        or get_nested(reports, ["trade_thesis", "conviction_level"])
    )
    conviction = conviction or get_nested(decision, ["conviction_level"])
    thesis_conviction = get_nested(reports, ["trade_thesis", "conviction_level"])
    if conviction and thesis_conviction and humanize_phrase(conviction) != humanize_phrase(thesis_conviction):
        conviction = thesis_conviction
    highlights: List[Dict[str, str]] = []
    proposal = summary_details
    underlying_price = proposal.get("underlying_price")
    if underlying_price is not None:
        highlights.append({"label": "Underlying Price", "value": format_currency(underlying_price)})
    expiration = proposal.get("expiration_date") or get_nested(decision, ["trade_legs", 0, "expiration_date"])
    if expiration:
        highlights.append({"label": "Target Expiration", "value": format_date(expiration)})
    computed_dte = summary.get("computed_dte")
    if computed_dte is None:
        computed_dte = _calculate_dte(expiration, decision.get("generated_at"))
    if computed_dte is not None:
        highlights.append({"label": "Days to Expiration", "value": f"{computed_dte} days"})
    holding = summary.get("holding_period")
    if holding and computed_dte is None:
        highlights.append({"label": "Days to Expiration (Agent)", "value": f"{holding} days"})
    cost = summary.get("cost_per_contract")
    if cost:
        highlights.append({"label": "Est. Cost / Contract", "value": format_currency(cost)})
    net_premium = proposal.get("net_premium")
    if net_premium is not None:
        label = "Net Credit" if (proposal.get("net_credit_debit") or "").lower().startswith("credit") else "Net Debit"
        highlights.append({"label": label, "value": format_currency(net_premium)})
    market_cap = derived.get("market_cap")
    if market_cap:
        highlights.append({"label": "Market Cap", "value": format_large_currency(market_cap)})

    # Get max_risk and max_reward from decision (primary source) or summary (fallback)
    max_risk = decision.get("max_risk") or summary.get("max_risk")
    max_reward = decision.get("max_reward") or summary.get("max_reward")

    # Get net_premium from decision if not already set
    if net_premium is None:
        net_premium = decision.get("net_premium")

    legs_source = decision.get("trade_legs") or proposal.get("trade_legs") or []
    legs: List[Dict[str, str]] = []
    for leg in legs_source:
        if not isinstance(leg, dict):
            continue
        greeks = leg.get("key_greeks_at_selection")
        greek_parts: List[str] = []
        if isinstance(greeks, dict):
            for name, value in greeks.items():
                if value is None:
                    continue
                label = name.replace("impliedVolatility", "IV")
                # FIX: Round to 2-3 decimals for readability, use HTML line breaks
                greek_parts.append(f"{humanize_phrase(label)}: <strong>{format_number(value, 2)}</strong>")
        legs.append(
            {
                "action": humanize_phrase(leg.get("action")),
                "type": humanize_phrase(leg.get("type")),
                "expiration": format_date(leg.get("expiration_date")),
                "strike": format_currency(leg.get("strike_price"), 2),
                "quantity": format_number(leg.get("quantity"), 0),
                "symbol": leg.get("contract_symbol") or "—",
                "greeks": "<br>".join(greek_parts) if greek_parts else "—",
            }
        )
    # Translate justification if Thai
    justification_text = summary.get("justification")
    if language == "th" and justification_text:
        justification_text = translate_to_thai(justification_text, "trade justification")
    justification_html = render_markdown_html(justification_text)
    action_display = summary_details.get("net_credit_debit") or decision.get("action") or proposal.get("action")
    if isinstance(action_display, str):
        action_display = humanize_phrase(action_display)
    else:
        action_display = humanize_phrase(decision.get("action"))
    quantity_value = decision.get("quantity") or summary_details.get("quantity")
    # Determine display values for risk/reward
    max_risk_display = format_currency(max_risk) if max_risk is not None else "N/A"

    # For max_reward, show "Unlimited" for strategies with unbounded profit (e.g. long calls/puts)
    if max_reward is not None:
        max_reward_display = format_currency(max_reward)
    elif max_risk is not None and max_risk > 0:
        # If we have defined risk but no defined reward, it's likely unlimited profit potential
        max_reward_display = "Unlimited"
    else:
        max_reward_display = "N/A"

    # Add capital required to highlights if available
    capital_required_amount = None
    capital_label = "Total Capital Required"

    if net_premium is not None:
        capital_required_amount = abs(net_premium)
        if capital_required_amount > 0:
            capital_label = "Capital Required" if net_premium < 0 else "Net Credit Received"
            highlights.insert(0, {
                "label": capital_label,
                "value": format_currency(capital_required_amount)
            })

    # Get strategy explanation for beginners (English or Thai based on language)
    strategy_name = humanize_phrase(strategy)
    if language == "th":
        strategy_info = STRATEGY_EXPLANATIONS_TH.get(strategy_name, {})
    else:
        strategy_info = STRATEGY_EXPLANATIONS.get(strategy_name, {})
    strategy_explanation = strategy_info.get("explanation")
    strategy_pros_cons = strategy_info.get("pros_cons")

    return {
        "direction": humanize_phrase(direction),
        "strategy": strategy_name,
        "conviction": humanize_phrase(conviction),
        "action": action_display,
        "quantity": format_number(quantity_value, 0) if quantity_value is not None else "1",
        "max_risk": max_risk_display,
        "max_reward": max_reward_display,
        "max_risk_display": max_risk_display,
        "max_reward_display": max_reward_display,
        "net_premium": format_currency(net_premium) if net_premium is not None else "N/A",
        "capital_required": format_currency(capital_required_amount) if capital_required_amount else None,
        "capital_label": capital_label,
        "strategy_explanation": strategy_explanation,
        "strategy_pros_cons": strategy_pros_cons,
        "highlights": highlights,
        "legs": legs,
        "justification_html": justification_html,
    }


def extract_headline(text: Optional[str]) -> str:
    if not text:
        return ""
    stripped = strip_code_blocks(text)
    stripped = stripped.strip()
    stripped = re.sub(r"^#+\s*", "", stripped)
    if not stripped:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    headline = sentences[0] if sentences else stripped
    headline = re.sub(r"^#+\s*", "", headline)
    return clean_text(headline)


def build_thesis_section(reports: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    thesis = reports.get("trade_thesis", {}) or {}
    summary_text = thesis.get("summary")
    payloads = extract_json_blocks(summary_text)
    primary_payload = payloads[0] if payloads and isinstance(payloads[0], dict) else {}
    narrative_plain = primary_payload.get("summary") if isinstance(primary_payload, dict) else None
    if not narrative_plain:
        narrative_plain = strip_code_blocks(summary_text)

    # Translate narrative if Thai
    if language == "th" and narrative_plain:
        narrative_plain = translate_to_thai(narrative_plain, "financial narrative")

    narrative_html = render_markdown_html(narrative_plain)
    evidence_items: List[Dict[str, str]] = []
    evidence_source = primary_payload.get("key_evidence") if isinstance(primary_payload, dict) else thesis.get("key_evidence")
    if isinstance(evidence_source, list):
        for item in evidence_source:
            if not isinstance(item, dict):
                continue
            detail_text = item.get("detail") or item.get("description")
            # Translate evidence detail if Thai
            if language == "th" and detail_text:
                detail_text = translate_to_thai(detail_text, "technical evidence")
            evidence_items.append(
                {
                    "type": humanize_phrase(item.get("type")),
                    "detail_html": render_markdown_html(detail_text),
                }
            )
    thesis_details: Dict[str, Any] = {}
    if isinstance(primary_payload, dict):
        thesis_details = {
            k: v
            for k, v in primary_payload.items()
            if k not in {"key_evidence", "summary"} and isinstance(v, (str, int, float, list, dict))
        }
    plan_rows = flatten_details(thesis_details, skip_keys={"trade_thesis", "conviction"}) if thesis_details else []
    conviction = humanize_phrase(primary_payload.get("conviction") or thesis.get("conviction_level"))
    headline = extract_headline(narrative_plain)
    # Translate headline if Thai
    if language == "th" and headline:
        headline = translate_to_thai(headline, "headline")
    return {
        "headline": headline,
        "narrative_html": narrative_html,
        "narrative_plain": narrative_plain or "",
        "evidence": evidence_items,
        "plan_rows": plan_rows,
        "conviction": conviction,
    }


def build_technical_section(reports: Dict[str, Any], derived: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    indicators = get_nested(reports, ["technical_report", "indicators"], {}) or {}
    rows: List[Dict[str, str]] = []
    latest_close = derived.get("latest_close")
    if latest_close is not None:
        context_note = f"Close on {derived.get('price_date_human')}" if derived.get("price_date_human") else "Latest close"
        rows.append({"label": "Latest Close", "value": format_currency(latest_close), "context": context_note})
    rsi_value = derived.get("rsi_value")
    if rsi_value is not None:
        rows.append(
            {
                "label": "RSI",
                "value": format_number(rsi_value, 1),
                "context": humanize_phrase(derived.get("rsi_regime")),
            }
        )
    macd_histogram = derived.get("macd_histogram")
    if macd_histogram is not None:
        rows.append(
            {
                "label": "MACD Histogram",
                "value": format_number(macd_histogram, 3),
                "context": humanize_phrase(derived.get("macd_crossover")),
            }
        )
    supertrend_level = derived.get("supertrend_level")
    if supertrend_level is not None:
        rows.append(
            {
                "label": "Supertrend Level",
                "value": format_currency(supertrend_level),
                "context": humanize_phrase(derived.get("supertrend_trend")),
            }
        )
    obv_value = derived.get("obv_value")
    if obv_value is not None:
        rows.append(
            {
                "label": "On-Balance Volume",
                "value": format_number(obv_value, 0),
                "context": humanize_phrase(derived.get("obv_trend")),
            }
        )
    for key in ("SMA_50", "SMA_200", "EMA_20"):
        node = indicators.get(key)
        if isinstance(node, dict):
            rows.append(
                {
                    "label": humanize_phrase(key),
                    "value": format_currency(node.get("value")),
                    "context": humanize_phrase(node.get("price_relationship")),
                }
            )
    bollinger = indicators.get("Bollinger_Bands")
    if isinstance(bollinger, dict):
        rows.append(
            {
                "label": "Bollinger Band Width",
                "value": format_number(bollinger.get("width"), 2),
                "context": f"Price position {format_percentage(bollinger.get('price_position'), 1)}",
            }
        )
    key_levels: List[Dict[str, str]] = []
    levels = indicators.get("key_levels") or {}
    if isinstance(levels, dict):
        for label, value in levels.items():
            key_levels.append({"label": humanize_phrase(label), "value": format_currency(value)})
    patterns: List[Dict[str, str]] = []
    for pattern in indicators.get("recent_candlestick_patterns", []) or []:
        if not isinstance(pattern, dict):
            continue
        patterns.append(
            {
                "date": format_date(pattern.get("date")),
                "pattern": humanize_phrase(pattern.get("pattern")),
                "signal": humanize_phrase(pattern.get("direction")),
            }
        )

    # Get and translate technical summary
    summary_text = get_nested(reports, ["technical_report", "llm_report", "summary"]) or ""
    if language == "th" and summary_text:
        summary_text = translate_to_thai(summary_text, "technical analysis")
    summary_html = render_markdown_html(summary_text)

    return {
        "bias": humanize_phrase(derived.get("technical_bias")),
        "summary_html": summary_html,
        "indicators": rows,
        "key_levels": key_levels,
        "patterns": patterns,
    }


def build_fundamental_section(reports: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    fundamental = reports.get("fundamental_report", {}) or {}

    # Get and translate business summary
    business_summary_text = get_nested(fundamental, ["business_overview", "longBusinessSummary"]) or ""
    if language == "th" and business_summary_text:
        business_summary_text = translate_to_thai(business_summary_text, "business description")
    business_summary_html = render_markdown_html(business_summary_text)

    ratios_list: List[Dict[str, str]] = []
    ratios = fundamental.get("financial_ratios") or {}
    if isinstance(ratios, dict):
        for key, value in ratios.items():
            ratios_list.append({"label": humanize_phrase(key), "value": format_number(value)})
    trend_tables: List[Dict[str, Any]] = []
    for trend in fundamental.get("financial_trends", []) or []:
        if not isinstance(trend, dict):
            continue
        values = trend.get("values")
        if not isinstance(values, dict):
            continue
        rows = [{"period": period, "value": format_large_currency(amount)} for period, amount in sorted(values.items())]
        trend_tables.append(
            {
                "metric": trend.get("metric", "Financial Trend"),
                "direction": humanize_phrase(trend.get("trend_direction")),
                "rows": rows,
            }
        )

    # Get MD&A summary and translate if needed
    mdna_summary_text = get_nested(fundamental, ["qualitative_summary", "mdna_summary", "summary"])
    mdna_struct = extract_json_blocks(mdna_summary_text)
    mdna_payload = mdna_struct[0] if mdna_struct and isinstance(mdna_struct[0], dict) else {}

    # Translate MD&A components if Thai
    tone = clean_text(get_nested(mdna_payload, ["tone"])) if mdna_payload else ""
    if language == "th" and tone:
        tone = translate_to_thai_simple(tone, "tone description")

    performance_drivers = [clean_text(item) for item in mdna_payload.get("performance_drivers", []) if isinstance(item, str)] if mdna_payload else []
    if language == "th":
        performance_drivers = [translate_to_thai_simple(item, "performance driver") for item in performance_drivers if item]

    opportunities = [clean_text(item) for item in get_nested(mdna_payload, ["forward_looking", "opportunities"], []) if isinstance(item, str)]
    if language == "th":
        opportunities = [translate_to_thai_simple(item, "opportunity") for item in opportunities if item]

    concerns = [clean_text(item) for item in get_nested(mdna_payload, ["forward_looking", "concerns"], []) if isinstance(item, str)]
    if language == "th":
        concerns = [translate_to_thai_simple(item, "concern") for item in concerns if item]

    # Translate MD&A narrative
    mdna_narrative_text = mdna_summary_text
    if language == "th" and mdna_narrative_text:
        mdna_narrative_text = translate_to_thai(mdna_narrative_text, "MD&A analysis")

    mdna = {
        "tone": tone,
        "performance_drivers": performance_drivers,
        "opportunities": opportunities,
        "concerns": concerns,
        "narrative_html": render_markdown_html(mdna_narrative_text),
    }

    risk_matrix: List[Dict[str, str]] = []
    risk_factors = get_nested(fundamental, ["qualitative_summary", "risk_factors"], [])
    if isinstance(risk_factors, list):
        for risk_entry in risk_factors:
            if not isinstance(risk_entry, dict):
                continue
            risk_text = risk_entry.get("risk")
            parsed_blocks = extract_json_blocks(risk_text)
            for block in parsed_blocks:
                if isinstance(block, list):
                    for item in block:
                        if not isinstance(item, dict):
                            continue
                        risk_name = clean_text(item.get("risk"))
                        risk_rationale = clean_text(item.get("rationale"))

                        # Translate risk components if Thai
                        if language == "th":
                            if risk_name:
                                risk_name = translate_to_thai_simple(risk_name, "risk factor")
                            if risk_rationale:
                                risk_rationale = translate_to_thai_simple(risk_rationale, "risk rationale")

                        risk_matrix.append(
                            {
                                "risk": risk_name,
                                "category": humanize_phrase(item.get("category")),
                                "rationale": risk_rationale,
                            }
                        )
                elif isinstance(block, dict):
                    risk_name = clean_text(block.get("risk"))
                    risk_rationale = clean_text(block.get("rationale"))

                    # Translate risk components if Thai
                    if language == "th":
                        if risk_name:
                            risk_name = translate_to_thai_simple(risk_name, "risk factor")
                        if risk_rationale:
                            risk_rationale = translate_to_thai_simple(risk_rationale, "risk rationale")

                    risk_matrix.append(
                        {
                            "risk": risk_name,
                            "category": humanize_phrase(block.get("category")),
                            "rationale": risk_rationale,
                        }
                    )
            break
    return {
        "business_summary_html": business_summary_html,
        "ratios": ratios_list,
        "trend_tables": trend_tables,
        "mdna": mdna,
        "risk_matrix": risk_matrix[:5],
    }


def build_volatility_section(reports: Dict[str, Any], derived: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    vol = reports.get("volatility_report", {}) or {}
    metrics = [
        {"label": "IV Rank", "value": format_number(vol.get("iv_rank"), 1) if vol.get("iv_rank") is not None else "N/A"},
        {"label": "Term Structure", "value": vol.get("term_structure", "N/A")},
        {"label": "Skew Analysis", "value": vol.get("skew_analysis", "N/A")},
    ]

    # Get and translate volatility forecast
    forecast = vol.get("volatility_forecast") or derived.get("volatility_forecast") or ""
    if language == "th" and forecast:
        forecast = translate_to_thai(forecast, "volatility forecast")

    return {"metrics": metrics, "forecast": forecast}


def build_sentiment_section(reports: Dict[str, Any], derived: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    sentiment = reports.get("sentiment_report", {}) or {}
    articles_output: List[Dict[str, str]] = []
    for article in sentiment.get("articles", [])[:5]:
        if not isinstance(article, dict):
            continue

        # Get article summary and translate if Thai
        article_summary = clean_text(article.get("rationale")) or "Summary unavailable."
        if language == "th" and article_summary and article_summary != "Summary unavailable.":
            article_summary = translate_to_thai_simple(article_summary, "news summary")

        articles_output.append(
            {
                "title": article.get("title", "Headline unavailable"),
                "publisher": article.get("publisher", "Unknown"),
                "published": format_timestamp(article.get("published_at")),
                "summary": article_summary,
                "link": article.get("link"),
            }
        )

    score = derived.get("sentiment_score")

    # Get and translate overall sentiment summary
    overall_summary = sentiment.get("overall_summary", derived.get("sentiment_summary") or "Neutral stance")
    if language == "th" and overall_summary:
        overall_summary = translate_to_thai_simple(overall_summary, "sentiment summary")

    return {
        "score": format_number(score, 1) if score is not None else "N/A",
        "summary": overall_summary,
        "articles": articles_output,
    }


def build_risk_section(reports: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    risk = reports.get("risk_assessment", {}) or {}
    adjustments: List[Dict[str, str]] = []
    for item in risk.get("adjustments", []) or []:
        if not isinstance(item, dict):
            continue

        # Get recommendation and translate if Thai
        recommendation = clean_text(item.get("recommendation"))
        if language == "th" and recommendation:
            recommendation = translate_to_thai_simple(recommendation, "risk recommendation")

        adjustments.append(
            {
                "profile": humanize_phrase(item.get("profile")),
                "recommendation": recommendation,
            }
        )

    # Get and translate final recommendation
    final_recommendation = risk.get("final_recommendation", "")
    if language == "th" and final_recommendation:
        final_recommendation = translate_to_thai(final_recommendation, "final risk assessment")

    return {"adjustments": adjustments, "final": final_recommendation}


# Backtest section removed - BacktesterAgent is no longer part of the pipeline
# def build_backtest_section(reports: Dict[str, Any]) -> Dict[str, Any]:
#     backtest = reports.get("backtest_report", {}) or {}
#     metrics: List[Dict[str, str]] = []
#     if isinstance(backtest, dict) and backtest:
#         metrics.append({
#             "label": "Win Rate",
#             "value": format_percentage(backtest.get("win_rate")),
#         })
#         metrics.append({
#             "label": "Simulated Trades",
#             "value": format_number(backtest.get("simulated_trades"), 0),
#         })
#         metrics.append({
#             "label": "Average Profit %",
#             "value": format_percentage(backtest.get("average_profit_pct")),
#         })
#     summary = backtest.get("summary") if isinstance(backtest, dict) else ""
#     strategy_type = backtest.get("strategy_type") if isinstance(backtest, dict) else ""
#     return {
#         "strategy_type": humanize_phrase(strategy_type),
#         "summary": summary or "Historical simulation not available for this run.",
#         "metrics": [item for item in metrics if item["value"] != "N/A"],
#     }


def build_visual_assets(reports: Dict[str, Any]) -> Dict[str, Any]:
    trends = get_nested(reports, ["fundamental_report", "financial_trends"], [])
    charts: List[Dict[str, str]] = []
    if not isinstance(trends, list):
        return {"trend_charts": charts}
    for trend in trends:
        if not isinstance(trend, dict):
            continue
        values = trend.get("values")
        if not isinstance(values, dict) or not values:
            continue
        try:
            points = sorted(
                (
                    (parse_iso_datetime(period) or datetime.fromisoformat(period), float(amount))
                    for period, amount in values.items()
                ),
                key=lambda pair: pair[0],
            )
        except (ValueError, TypeError):
            continue
        if not points:
            continue
        dates, series = zip(*points)
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=160)
        ax.plot(dates, series, color="#1D4ED8", linewidth=2.0, marker="o", markersize=4)
        ax.fill_between(dates, series, color="#1D4ED8", alpha=0.12)
        ax.set_title(trend.get("metric", "Financial Trend"), fontsize=10, color="#0F172A", pad=10)
        ax.grid(color="#CBD5F5", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.tick_params(colors="#475569", labelsize=8)
        fig.autofmt_xdate()
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        charts.append(
            {
                "label": trend.get("metric", "Financial Trend"),
                "image": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}",
            }
        )
        break
    return {"trend_charts": charts}


def build_executive_summary(
    trade_section: Dict[str, Any],
    derived: Dict[str, Any],
    thesis_section: Dict[str, Any],
    fundamentals: Dict[str, Any],
    volatility: Dict[str, Any],
    language: str = "en",
) -> Dict[str, Any]:
    bullets: List[str] = []
    direction = trade_section.get("direction")
    strategy = trade_section.get("strategy")
    conviction = trade_section.get("conviction")
    thesis_conviction = thesis_section.get("conviction") if isinstance(thesis_section, dict) else None
    if thesis_conviction is None:
        thesis_conviction = get_nested(fundamentals, ["trade_thesis", "conviction_level"])
    if thesis_conviction and conviction and humanize_phrase(conviction) != humanize_phrase(thesis_conviction):
        conviction = humanize_phrase(thesis_conviction)
    if direction or strategy:
        conviction_text = f" ({conviction} conviction)" if conviction else ""
        bullets.append(clean_text(f"{direction or 'Trade'} {strategy or ''}{conviction_text}"))
    bullets.append(
        clean_text(
            f"Risk envelope: {trade_section.get('max_risk')} max risk / {trade_section.get('max_reward')} potential reward."
        )
    )
    technical_bias = derived.get("technical_bias")
    if technical_bias or derived.get("rsi_value") is not None:
        bias_text = humanize_phrase(technical_bias) or "Neutral"
        rsi_text = format_number(derived.get("rsi_value"), 1) if derived.get("rsi_value") is not None else "N/A"
        macd_context = humanize_phrase(derived.get("macd_crossover")) or "steady"
        bullets.append(f"Technical view: {bias_text} bias with RSI {rsi_text} and {macd_context} MACD crossover.")
    mdna_tone = fundamentals.get("mdna", {}).get("tone")
    if mdna_tone:
        bullets.append(f"Management tone: {mdna_tone}")
    if volatility.get("forecast"):
        bullets.append(volatility["forecast"])
    headline = clean_text(thesis_section.get("headline") or direction or "Trade Summary")
    headline = re.sub(r"^#+\s*", "", headline).strip()
    bullets = [clean_text(item) for item in bullets if clean_text(item)]

    # Translate bullets and headline if Thai
    if language == "th":
        bullets = [translate_to_thai(bullet, "executive summary") for bullet in bullets if bullet]
        if headline:
            headline = translate_to_thai(headline, "headline")

    return {"headline": headline, "bullets": bullets}


def build_meta_items(
    ticker: str,
    timestamp: str,
    report_date: str,
    results_folder: str,
    trade: Dict[str, Any],
    volatility: Dict[str, Any],
    sentiment: Dict[str, Any],
) -> List[Dict[str, str]]:
    return [
        {"label": "Ticker", "value": ticker.upper()},
        {"label": "Results Folder", "value": results_folder},
        {"label": "Run Timestamp", "value": timestamp},
        {"label": "Report Generated", "value": report_date},
        {"label": "Max Risk", "value": trade.get("max_risk", "N/A")},
        {"label": "Potential Reward", "value": trade.get("max_reward", "N/A")},
        {"label": "Volatility Forecast", "value": volatility.get("forecast") or "Unavailable"},
        {"label": "Sentiment Score", "value": sentiment.get("score", "N/A")},
    ]


def build_report_context(reports: Dict[str, Any], ticker: str, timestamp: str, results_path: str, language: str = "en") -> Dict[str, Any]:
    derived = build_derived_fields(reports)
    trade_summary = extract_trade_summary(reports.get("trade_decision", {}) or {})
    reports["trade_summary"] = trade_summary
    # FIX: Pass timestamp to prefer folder name date over JSON dates
    report_dt = infer_report_datetime(reports, timestamp=timestamp)

    # Build all sections with language parameter for translation
    thesis_section = build_thesis_section(reports, language=language)
    trade_section = build_trade_section(reports, derived, trade_summary, language=language)
    technical_section = build_technical_section(reports, derived, language=language)
    fundamentals = build_fundamental_section(reports, language=language)
    volatility = build_volatility_section(reports, derived, language=language)
    sentiment = build_sentiment_section(reports, derived, language=language)
    risk_section = build_risk_section(reports, language=language)
    visuals = build_visual_assets(reports)
    executive_summary = build_executive_summary(trade_section, derived, thesis_section, fundamentals, volatility, language=language)

    market_snapshot_rows = build_market_snapshot_rows(derived)
    report_date = format_timestamp(report_dt)
    results_folder = os.path.basename(results_path)
    meta_items = build_meta_items(ticker, timestamp, report_date, results_folder, trade_section, volatility, sentiment)
    return {
        "ticker": ticker.upper(),
        "timestamp": timestamp,
        "report_date": report_date,
        "results_folder": results_folder,
        "meta": meta_items,
        "market_snapshot_rows": market_snapshot_rows,
        "executive_summary": executive_summary,
        "trade": trade_section,
        "thesis": thesis_section,
        "technical": technical_section,
        "fundamentals": fundamentals,
        "volatility": volatility,
        "sentiment": sentiment,
        "risk": risk_section,
        "visuals": visuals,
        "disclaimer": "This report is prepared by the Quant13 multi-agent research framework and does not constitute investment advice.",
    }


def compose_llm_sections(context: Dict[str, Any], reports: Dict[str, Any]) -> Dict[str, Any]:
    llm_client = get_llm_client()
    trade = context.get("trade", {}) or {}
    payload = {
        "ticker": context.get("ticker"),
        "executive_summary": context.get("executive_summary"),
        "trade": {
            "strategy": trade.get("strategy"),
            "direction": trade.get("direction"),
            "conviction": trade.get("conviction"),
            "highlights": trade.get("highlights"),
        },
        "risk": context.get("risk"),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial editor. Craft polished HTML sections for an options research report. "
                "Respond with JSON containing 'hero_html' and 'sections'."
            ),
        },
        {"role": "user", "content": json.dumps(payload)},
    ]

    parsed: Optional[Dict[str, Any]] = None
    try:
        response = llm_client.chat(messages, temperature=0.25)
        parsed = json.loads(response)
    except Exception:
        parsed = None

    content = _normalize_llm_sections(parsed)
    if content is None:
        content = _default_llm_sections(context)
    return {"content": content}


def _normalize_llm_sections(candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, dict):
        return None
    hero = candidate.get("hero_html")
    sections = candidate.get("sections")
    if not isinstance(hero, str) or not hero.strip():
        return None
    normalized_sections: List[Dict[str, str]] = []
    if isinstance(sections, list):
        for entry in sections:
            if not isinstance(entry, dict):
                continue
            title = entry.get("title")
            html = entry.get("html")
            if isinstance(title, str) and isinstance(html, str):
                normalized_sections.append({"title": title, "html": html})
    return {"hero_html": hero, "sections": normalized_sections}


def _default_llm_sections(context: Dict[str, Any]) -> Dict[str, Any]:
    ticker = context.get("ticker") or "Ticker"
    trade = context.get("trade", {}) or {}
    executive = context.get("executive_summary", {}) or {}
    headline = executive.get("headline") or f"Options Outlook – {ticker}"
    conviction = humanize_phrase(trade.get("conviction")) or "Not specified"
    strategy = humanize_phrase(trade.get("strategy")) or "No strategy provided"
    direction = humanize_phrase(trade.get("direction")) or "Neutral"
    hero_html = (
        f"<section><h1>Options Outlook – {ticker}</h1>"
        f"<p>{clean_text(headline)}</p></section>"
    )
    sections: List[Dict[str, str]] = [
        {
            "title": "Trade Overview",
            "html": (
                f"<p><strong>Strategy:</strong> {strategy}<br />"
                f"<strong>Direction:</strong> {direction}<br />"
                f"<strong>Conviction:</strong> {conviction}</p>"
            ),
        }
    ]
    # FIX: backtest variable is not defined in this function - removed backtest section
    # Backtest is no longer part of the pipeline (BacktesterAgent removed)
    # If re-enabled, retrieve from context: backtest = context.get("backtest_report", {})

    # backtest_summary = backtest.get("summary") if isinstance(backtest, dict) else None
    # if backtest_summary:
    #     sections.append({
    #         "title": "Historical Simulation",
    #         "html": f"<p>{clean_text(backtest_summary)}</p>",
    #     })
    risk_final = context.get("risk", {}).get("final") if isinstance(context.get("risk"), dict) else None
    if risk_final:
        sections.append({
            "title": "Risk Desk Guidance",
            "html": f"<p>{clean_text(risk_final)}</p>",
        })
    return {"hero_html": hero_html, "sections": sections}


def render_html(context: Dict[str, Any]) -> str:
    template_dir = os.path.join(BASE_DIR, "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report_template.html")
    return template.render(context=context)


def create_pdf(html_string: str, output_file: str) -> None:
    base_url = os.path.join(BASE_DIR, "templates")
    HTML(string=html_string, base_url=base_url).write_pdf(output_file)


def build_output_path(base_dir: str, ticker: str, timestamp: str, output: Optional[str], language: str = "en") -> str:
    if output:
        return output
    lang_suffix = f"_{language}" if language != "en" else ""
    filename = f"{ticker.upper()}_{timestamp}_report{lang_suffix}.pdf"
    return os.path.join(base_dir, filename)


def generate_pdf_report(
    ticker: str,
    timestamp: str,
    base_results_dir: Optional[str] = None,
    output: Optional[str] = None,
    language: str = "en",
) -> str:
    base_dir = base_results_dir or os.path.join(BASE_DIR, "results")
    results_path = os.path.join(base_dir, f"{ticker.upper()}_{timestamp}")
    if not os.path.isdir(results_path):
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    reports = load_all_reports(results_path)
    context = build_report_context(reports, ticker, timestamp, results_path, language=language)

    html = render_html(context)
    output_path = build_output_path(results_path, ticker, timestamp, output, language=language)
    create_pdf(html, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Quant13 PDF trading report.")
    parser.add_argument("ticker", help="Ticker symbol used for the run")
    parser.add_argument(
        "timestamp",
        help="Timestamp suffix used in the results directory (e.g. YYYYMMDD_HHMMSS)",
    )
    parser.add_argument("--output", help="Optional output PDF path")
    args = parser.parse_args()

    output_path = generate_pdf_report(args.ticker, args.timestamp, output=args.output)
    print(f"PDF report generated at: {output_path}")


if __name__ == "__main__":
    main()
