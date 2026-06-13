import logging
import re

import pixeltable as pxt

logger = logging.getLogger(__name__)


@pxt.udf
def create_messages(image_raw: str) -> list[dict]:
    return [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': image_raw}},
                {'type': 'text', 'text': 'Please analyze this trading chart:'},
            ],
        }
    ]


@pxt.udf
def parse_support_resistance(analysis: str) -> dict[str, list[float | None]]:
    try:
        support_levels: list[float | None] = [None, None, None]
        resistance_levels: list[float | None] = [None, None, None]

        for level_num, value in re.findall(r'S(\d)\s*=\s*\$?([\d.]+)', analysis):
            idx = int(level_num) - 1
            if 0 <= idx < 3:
                try:
                    support_levels[idx] = float(value)
                except ValueError:
                    continue

        for level_num, value in re.findall(r'R(\d)\s*=\s*\$?([\d.]+)', analysis):
            idx = int(level_num) - 1
            if 0 <= idx < 3:
                try:
                    resistance_levels[idx] = float(value)
                except ValueError:
                    continue

        return {'support': support_levels, 'resistance': resistance_levels}
    except Exception as e:
        logger.error(f'Error in parse_support_resistance: {e}')
        return {'support': [None, None, None], 'resistance': [None, None, None]}


@pxt.udf
def extract_technical_indicators(analysis: str) -> dict:
    try:
        indicators = {
            'macd': '-',
            'rsi': '-',
            'mfi': '-',
            'stochastic': '-',
            'volume': '-',
            'current_price': '-',
            'vwap': '-',
        }
        for line in analysis.split('\n'):
            if 'MACD:' in line:
                indicators['macd'] = line.split('MACD:', 1)[1].strip()
            elif 'RSI:' in line:
                indicators['rsi'] = line.split('RSI:', 1)[1].strip()
            elif '(MFI):' in line:
                indicators['mfi'] = line.split('(MFI):', 1)[1].strip()
            elif 'Stochastic:' in line:
                indicators['stochastic'] = line.split('Stochastic:', 1)[1].strip()
            elif 'Volume:' in line:
                indicators['volume'] = line.split('Volume:', 1)[1].strip()
            elif 'Current Price:' in line:
                indicators['current_price'] = line.split('Current Price:', 1)[1].strip()
            elif 'VWAP:' in line:
                indicators['vwap'] = line.split('VWAP:', 1)[1].strip()
        return indicators
    except Exception as e:
        logger.error(f'Error parsing technical indicators: {e}')
        return {
            'macd': '-',
            'rsi': '-',
            'mfi': '-',
            'stochastic': '-',
            'volume': '-',
            'current_price': '-',
            'vwap': '-',
        }


@pxt.udf
def extract_trade_setup(analysis: str) -> dict:
    try:
        trade_setup = {'entry': '-', 'stop_loss': '-', 'target': '-'}
        for line in analysis.split('\n'):
            if 'Entry:' in line:
                trade_setup['entry'] = line.split(':', 1)[1].strip()
            elif 'Stop Loss:' in line:
                trade_setup['stop_loss'] = line.split(':', 1)[1].strip()
            elif 'Target:' in line:
                trade_setup['target'] = line.split(':', 1)[1].strip()
        return trade_setup
    except Exception as e:
        logger.error(f'Error parsing trade setup: {e}')
        return {'entry': '-', 'stop_loss': '-', 'target': '-'}


@pxt.udf
def extract_signal_type(analysis: str) -> str:
    try:
        analysis_upper = analysis.upper()
        for pattern in [
            r'Signal.*?:\s*(BULLISH|BEARISH|NEUTRAL)',
            r'Signal Confidence.*?:\s*(BULLISH|BEARISH|NEUTRAL)',
            r'Overall.*?(BULLISH|BEARISH|NEUTRAL)',
            r'(BULLISH|BEARISH|NEUTRAL)\s+(?:signal|bias|trend)',
        ]:
            match = re.search(pattern, analysis_upper)
            if match:
                signal = match.group(1).strip()
                if signal in ['BULLISH', 'BEARISH', 'NEUTRAL']:
                    return signal

        sentiment_matches = {
            'BULLISH': len(re.findall(r'\b(BULLISH|UPTREND|LONG|CALL)\b', analysis_upper)),
            'BEARISH': len(re.findall(r'\b(BEARISH|DOWNTREND|SHORT|PUT)\b', analysis_upper)),
            'NEUTRAL': len(re.findall(r'\b(NEUTRAL|SIDEWAYS|CONSOLIDATING)\b', analysis_upper)),
        }
        max_sentiment = max(sentiment_matches.items(), key=lambda x: x[1])
        if max_sentiment[1] > 0:
            return max_sentiment[0]
        return 'NEUTRAL'
    except Exception as e:
        logger.error(f'Error extracting signal type: {e}')
        return 'NEUTRAL'


@pxt.udf
def parse_summary(analysis: str) -> str:
    try:
        for pattern in [
            r'Summary\s*[:=-]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Summary:\s*(.*?)(?=\n\d|\n[A-Z]|$)',
            r'Overall Analysis\s*[:=-]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'\*\*Summary\*\*:?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        ]:
            match = re.search(pattern, analysis, re.DOTALL | re.IGNORECASE)
            if match:
                summary = re.sub(r'\s+', ' ', match.group(1).strip()).strip('*- ')
                if summary and len(summary) > 5:
                    return summary
        return 'Analysis summary not available.'
    except Exception as e:
        logger.error(f'Error parsing summary: {e}')
        return 'Error retrieving analysis summary.'
