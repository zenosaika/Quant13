"""
Discord webhook integration for sending trade reports.
"""
import os
import json
import requests
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


class DiscordReportSender:
    """Sends formatted trade reports to Discord via webhook."""

    def __init__(self, webhook_url: str):
        """
        Initialize Discord sender.

        Args:
            webhook_url: Discord webhook URL
        """
        self.webhook_url = webhook_url

    def send_trade_report(
        self,
        ticker: str,
        result_dir: Path,
        trade_decision: Dict[str, Any],
        trade_thesis: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        pdf_path: Optional[Path] = None
    ) -> bool:
        """
        Send a formatted trade report to Discord.

        Args:
            ticker: Stock ticker symbol
            result_dir: Path to results directory
            trade_decision: Trade decision data
            trade_thesis: Trade thesis data
            risk_assessment: Risk assessment data
            pdf_path: Optional path to PDF report

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build the Discord message payload
            payload = self._build_payload(
                ticker,
                trade_decision,
                trade_thesis,
                risk_assessment
            )

            # If PDF exists, send it with the embed in the same message
            if pdf_path and pdf_path.exists():
                print(f"📎 Attaching PDF: {pdf_path.name}")
                with open(pdf_path, 'rb') as f:
                    files = {
                        'file': (f'Quant13_{ticker}_Report.pdf', f, 'application/pdf')
                    }
                    # Send embed with PDF attachment
                    response = requests.post(
                        self.webhook_url,
                        data={'payload_json': json.dumps(payload)},
                        files=files
                    )

                    if response.status_code not in [200, 204]:
                        print(f"❌ Discord webhook failed: {response.status_code} - {response.text}")
                        return False

                    print(f"✅ Message and PDF sent successfully!")
            else:
                # Send without PDF
                print(f"⚠️  No PDF found, sending message only")
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code not in [200, 204]:
                    print(f"❌ Discord webhook failed: {response.status_code} - {response.text}")
                    return False

            return True

        except Exception as e:
            print(f"❌ Error sending to Discord: {str(e)}")
            return False

    def _build_payload(
        self,
        ticker: str,
        trade_decision: Dict[str, Any],
        trade_thesis: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the Discord webhook payload with rich embeds."""

        # Extract key data from correct JSON structure
        strategy_type = trade_decision.get("strategy_name", "UNKNOWN")
        action = trade_decision.get("action", "HOLD")
        thesis_type = trade_thesis.get("winning_argument", "NEUTRAL").upper()
        conviction_level = trade_thesis.get("conviction_level", "Unknown")

        # Convert conviction to confidence score
        confidence_map = {"High": 0.85, "Medium": 0.65, "Low": 0.45, "Unknown": 0.0}
        confidence = confidence_map.get(conviction_level, 0.0)

        # Determine color based on thesis
        color_map = {
            "BULLISH": 0x00FF00,      # Green
            "BEARISH": 0xFF0000,       # Red
            "NEUTRAL": 0xFFFF00,       # Yellow
            "MIXED": 0xFFA500          # Orange
        }
        embed_color = color_map.get(thesis_type, 0x808080)

        # Get risk metrics from trade_decision
        max_risk = trade_decision.get("max_risk") or 0
        max_reward = trade_decision.get("max_reward") or 0

        # Derive risk rating from max_risk and conviction
        if max_risk == 0:
            risk_rating = "LOW"
        elif max_risk < 1000:
            risk_rating = "LOW"
        elif max_risk < 3000:
            risk_rating = "MEDIUM"
        elif max_risk < 10000:
            risk_rating = "HIGH"
        else:
            risk_rating = "EXTREME"

        # Build position details
        legs = trade_decision.get("trade_legs", [])
        position_desc = self._format_position(legs, strategy_type)

        # Create sentiment emoji
        sentiment_emoji = {
            "BULLISH": "🐂",
            "BEARISH": "🐻",
            "NEUTRAL": "😐",
            "MIXED": "🤔"
        }.get(thesis_type, "❓")

        # Action emoji - handle both simple and compound actions
        action_emoji = "⏸️"  # Default
        if "BUY" in action.upper():
            action_emoji = "💰"
        elif "SELL" in action.upper():
            action_emoji = "💸"
        elif "HOLD" in action.upper():
            action_emoji = "✋"

        # Risk emoji
        risk_emoji = {
            "LOW": "🟢",
            "MEDIUM": "🟡",
            "HIGH": "🔴",
            "EXTREME": "⚠️"
        }.get(risk_rating, "⚪")

        # Build the main embed
        embed = {
            "title": f"🎯 OPTIONS TRADE ALERT: ${ticker}",
            "description": f"**War Room Analysis Complete** {sentiment_emoji}\n\n*The debate has concluded. Our multi-agent system has reached a verdict.*",
            "color": embed_color,
            "fields": [
                {
                    "name": f"{action_emoji} Recommended Action",
                    "value": f"**{action}** - {strategy_type.replace('_', ' ').title()}",
                    "inline": True
                },
                {
                    "name": f"{sentiment_emoji} Market Thesis",
                    "value": f"**{thesis_type}** ({confidence:.0%} confidence)",
                    "inline": True
                },
                {
                    "name": f"{risk_emoji} Risk Rating",
                    "value": f"**{risk_rating}**",
                    "inline": True
                },
                {
                    "name": "📊 Position Structure",
                    "value": position_desc,
                    "inline": False
                },
                {
                    "name": "💵 Risk/Reward Profile",
                    "value": f"**Max Risk:** ${max_risk:,.2f}\n**Max Reward:** ${max_reward:,.2f}\n**R/R Ratio:** {self._calculate_rr_ratio(max_risk, max_reward)}",
                    "inline": False
                },
                {
                    "name": "🧠 Executive Summary",
                    "value": self._clean_summary(trade_thesis.get("summary", "Analysis complete. See PDF for full details."), 500),
                    "inline": False
                }
            ],
            "footer": {
                "text": f"🤖 Quant13 Multi-Agent Options System | Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | NOT FINANCIAL ADVICE"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add warning if high risk
        if risk_rating in ["HIGH", "EXTREME"]:
            embed["fields"].insert(0, {
                "name": "⚠️ RISK WARNING",
                "value": "This trade carries significant risk. Review full analysis before proceeding.",
                "inline": False
            })

        # Build payload
        payload = {
            "content": f"@here **New {ticker} Options Analysis Ready!** 📈\n\n",
            "embeds": [embed],
            "username": "Quant13 War Room",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/3135/3135706.png"  # Robot icon
        }

        return payload

    def _format_position(self, legs: list, strategy_type: str) -> str:
        """Format the options position for display."""
        if not legs:
            return f"*{strategy_type.replace('_', ' ').title()} Strategy*"

        formatted_legs = []
        for leg in legs[:4]:  # Limit to 4 legs for space
            action = leg.get("action", "")
            quantity = leg.get("quantity", 0)
            option_type = leg.get("type", "")  # Changed from option_type to type
            strike = leg.get("strike_price", 0)  # Changed from strike to strike_price
            expiration = leg.get("expiration_date", "")  # Changed from expiration to expiration_date

            leg_str = f"`{action} {quantity}x {option_type} ${strike} exp {expiration}`"
            formatted_legs.append(leg_str)

        if len(legs) > 4:
            formatted_legs.append(f"*...and {len(legs) - 4} more leg(s)*")

        return "\n".join(formatted_legs) if formatted_legs else f"*{strategy_type} structure*"

    def _calculate_rr_ratio(self, max_risk: float, max_reward: float) -> str:
        """Calculate and format risk/reward ratio."""
        if max_risk <= 0:
            return "N/A"
        ratio = max_reward / max_risk
        return f"1:{ratio:.2f}"

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    def _clean_summary(self, text: str, max_length: int = 400) -> str:
        """Clean and format summary text for Discord display."""
        import re

        # Remove JSON code blocks
        text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

        # Remove markdown headers (##, ###, etc.)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        # Remove extra whitespace and newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        # Split into sentences and take first few
        sentences = []
        current_length = 0

        # Split by periods followed by space or newline
        for sentence in re.split(r'\.(?:\s+|\n)', text):
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_with_period = sentence + '.'
            if current_length + len(sentence_with_period) > max_length:
                break

            sentences.append(sentence_with_period)
            current_length += len(sentence_with_period) + 1

        result = ' '.join(sentences)

        # If still too long, truncate
        if len(result) > max_length:
            result = result[:max_length - 3] + "..."

        return result if result else "Analysis complete. See PDF for full details."

    def _send_pdf(self, pdf_path: Path, ticker: str) -> bool:
        """Send PDF as a file attachment."""
        try:
            with open(pdf_path, 'rb') as f:
                files = {
                    'file': (f'Quant13_{ticker}_Report.pdf', f, 'application/pdf')
                }

                payload = {
                    "content": f"📄 **Full Analysis Report Attached**\n\n*Detailed multi-agent analysis, charts, and risk breakdowns for ${ticker}*"
                }

                response = requests.post(
                    self.webhook_url,
                    data=payload,
                    files=files
                )

                return response.status_code in [200, 204]

        except Exception as e:
            print(f"❌ Error sending PDF: {str(e)}")
            return False


def send_to_discord(
    ticker: str,
    result_dir: Path,
    webhook_url: Optional[str] = None
) -> bool:
    """
    Convenience function to send results to Discord.

    Args:
        ticker: Stock ticker symbol
        result_dir: Path to results directory
        webhook_url: Optional webhook URL (uses env var if not provided)

    Returns:
        True if successful, False otherwise
    """
    # Get webhook URL from environment if not provided
    if not webhook_url:
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        print("⚠️  No Discord webhook URL configured. Skipping Discord notification.")
        return False

    # Load required JSON files
    try:
        with open(result_dir / "trade_decision.json", "r") as f:
            trade_decision = json.load(f)

        with open(result_dir / "trade_thesis.json", "r") as f:
            trade_thesis = json.load(f)

        with open(result_dir / "risk_assessment.json", "r") as f:
            risk_assessment = json.load(f)

        # Check for PDF (try both naming patterns)
        pdf_path = result_dir / "report.pdf"
        if not pdf_path.exists():
            # Try timestamped format: TICKER_YYYYMMDD_HHMMSS_report.pdf
            pdf_files = list(result_dir.glob("*_report.pdf"))
            pdf_path = pdf_files[0] if pdf_files else None

        # Send to Discord
        sender = DiscordReportSender(webhook_url)
        success = sender.send_trade_report(
            ticker,
            result_dir,
            trade_decision,
            trade_thesis,
            risk_assessment,
            pdf_path
        )

        if success:
            print(f"✅ Report sent to Discord successfully!")

        return success

    except Exception as e:
        print(f"❌ Failed to send Discord notification: {str(e)}")
        return False
