import base64
import getpass
import io
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

import pixeltable as pxt
from pixeltable.functions import image as pxt_image
from pixeltable.functions.anthropic import messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

if 'ANTHROPIC_API_KEY' not in os.environ:
    os.environ['ANTHROPIC_API_KEY'] = getpass.getpass('ANTHROPIC_API_KEY:')

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*']
)

pxt.drop_dir('trade_analysis', force=True)
pxt.create_dir('trade_analysis')

analysis_table = pxt.create_table(
    'trade_analysis.analysis', {'image': pxt.Image, 'timestamp': pxt.Timestamp, 'request_id': pxt.String}
)

analysis_table.add_computed_column(image_raw=pxt_image.b64_encode(analysis_table.image, 'jpeg'))


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
def parse_support_resistance(analysis: str) -> dict[str, list[Optional[float]]]:
    try:
        support_levels = [None, None, None]
        resistance_levels = [None, None, None]

        s_pattern = r'S(\d)\s*=\s*\$?([\d.]+)'
        r_pattern = r'R(\d)\s*=\s*\$?([\d.]+)'

        s_matches = re.findall(s_pattern, analysis)
        for level_num, value in s_matches:
            idx = int(level_num) - 1
            if 0 <= idx < 3:
                try:
                    support_levels[idx] = float(value)
                except ValueError:
                    continue

        r_matches = re.findall(r_pattern, analysis)
        for level_num, value in r_matches:
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
        lines = analysis.split('\n')
        indicators = {
            'macd': '-',
            'rsi': '-',
            'mfi': '-',
            'stochastic': '-',
            'volume': '-',
            'current_price': '-',
            'vwap': '-',
        }

        for line in lines:
            if 'MACD:' in line:
                indicators['macd'] = line.lstrip('MACD:').strip()
            elif 'RSI:' in line:
                indicators['rsi'] = line.lstrip('RSI:').strip()
            elif '(MFI):' in line:
                indicators['mfi'] = line.lstrip('(MFI):').strip()
            elif 'Stochastic:' in line:
                indicators['stochastic'] = line.lstrip('Stochastic:').strip()
            elif 'Volume:' in line:
                indicators['volume'] = line.lstrip('Volume:').strip()
            elif 'Current Price:' in line:
                indicators['current_price'] = line.lstrip('Current Price:').strip()
            elif 'VWAP:' in line:
                indicators['vwap'] = line.lstrip('VWAP:').strip()

        logger.info(f'Extracted indicators: {indicators}')
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
        lines = analysis.split('\n')
        trade_setup = {'entry': '-', 'stop_loss': '-', 'target': '-'}

        for line in lines:
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
        signal_patterns = [
            r'Signal.*?:\s*(BULLISH|BEARISH|NEUTRAL)',
            r'Signal Confidence.*?:\s*(BULLISH|BEARISH|NEUTRAL)',
            r'Overall.*?(BULLISH|BEARISH|NEUTRAL)',
            r'(BULLISH|BEARISH|NEUTRAL)\s+(?:signal|bias|trend)',
        ]

        analysis_upper = analysis.upper()

        for pattern in signal_patterns:
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
        summary_patterns = [
            r'Summary\s*[:=-]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Summary:\s*(.*?)(?=\n\d|\n[A-Z]|$)',
            r'Overall Analysis\s*[:=-]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'\*\*Summary\*\*:?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        ]

        for pattern in summary_patterns:
            match = re.search(pattern, analysis, re.DOTALL | re.IGNORECASE)
            if match:
                summary = match.group(1).strip()
                if summary:
                    summary = re.sub(r'\s+', ' ', summary)
                    summary = summary.strip('*- ')
                    return summary if len(summary) > 5 else 'Summary not available.'

        sections = analysis.split('\n\n')
        for section in reversed(sections):
            if 'summary' in section.lower() or 'overall' in section.lower():
                summary = re.sub(r'^.*?[:=-]\s*', '', section, flags=re.IGNORECASE)
                summary = re.sub(r'\s+', ' ', summary).strip()
                if summary and len(summary) > 5:
                    return summary

        return 'Analysis summary not available.'
    except Exception as e:
        logger.error(f'Error parsing summary: {e}')
        return 'Error retrieving analysis summary.'


analysis_table.add_computed_column(messages=create_messages(analysis_table.image_raw))

analysis_table.add_computed_column(
    claude_response=messages(
        model='claude-3-sonnet-20240229',
        messages=analysis_table.messages,
        max_tokens=1500,
        temperature=0.1,
        top_p=0.3,
        top_k=10,
        system="""
You are a professional market analyst specializing in technical analysis for day trading. Analyze trading charts with precision and provide structured insights to determine current trends leveraging MACD, RSI, Money Flow Index, EMA, SMA, Bollinger Bands, Stochastic oscillator, VWAP, Volume for options trading.

CHART ANALYSIS METHODOLOGY:
1. Time Frame Analysis
- On multi-timeframe charts: Primary analysis on 5-minute, validate with 1-minute, confirm with 30-second and specify analysis per charts if relevant.
- Focus on recent swings for support/resistance levels
- Factor in pre-market and post-market movements if visible
2. Technical Analysis Rules:
MACD Interpretation:
- Blue line (MACD): Primary trend indicator
- Orange line (Signal): Confirmation line
- Green/Red histogram: Momentum indicator
- Look for: Crossovers, divergences, histogram changes
RSI & Stochastic Guidelines:
- RSI: Overbought >70, Oversold <30
- Stochastic: Overbought >80, Oversold <20
- %K (green) crossing %D (red) signals potential reversals
- Look for divergences with price
Pattern Recognition:
For BULLISH Setups:
- Price breaking above resistance
- Supporting volume increase
- MACD positive divergence
- RSI/Stochastic confirming momentum
For BEARISH Setups:
- Price breaking below support
- MACD negative divergence
- Bearish stochastic crossover
- Volume confirmation
Risk Management:
- No trades in first 5 minutes (6:30-6:35 AM PT)
- Minimum 1:1.5 risk-reward ratio
- Stop loss beyond nearest support/resistance
- Volume must confirm price action

REQUIRED OUTPUT FORMAT:
1. Current Trend:
ONE LINE trend description
2. Key Price Levels:
Support Levels: S1 = $X.XX, S2 = $X.XX, S3 = $X.XX
Resistance Levels:  R1 = $X.XX, R2 = $X.XX, R3 = $X.XX
Current Price: $X.XX
VWAP: $X.XX

3. Technical Indicators:
MACD:  value = X.XX, signal = X.XX, histogram = X.XX, with interpretation
RSI: X.XX, with interpretation
Money flow index (MFI): X.XX, with interpretation
Stochastic: %K = X.XX, %D = X.XX, with interpretation
Volume: ONE LINE trend analysis
4. Trade Setup:
Entry:  $X.XX, direction: PUT/CALL, with interpretation
Stop Loss:  $X.XX, with interpretation
Target:  $X.XX, with interpretation

5. Signal Type: BULLISH, BEARISH, or NEUTRAL
6. Summary: ONE sentence combining key signals and setup
FORMAT RULES:
- All prices in $X.XX format
- Support/Resistance labeled as S1/S2/S3 and R1/R2/R3
- Missing data marked as "Not Available"
- Signal Type must be  BULLISH, BEARISH, or NEUTRAL
- Each section must follow exact format for reliable parsing""",
    )
)

analysis_table.add_computed_column(analysis_text=analysis_table.claude_response.content[0].text)

analysis_table.add_computed_column(levels=parse_support_resistance(analysis_table.analysis_text))

analysis_table.add_computed_column(indicators=extract_technical_indicators(analysis_table.analysis_text))

analysis_table.add_computed_column(trade_setup=extract_trade_setup(analysis_table.analysis_text))

analysis_table.add_computed_column(signal_type=extract_signal_type(analysis_table.analysis_text))

analysis_table.add_computed_column(summary=parse_summary(analysis_table.analysis_text))


class ScreenshotRequest(BaseModel):
    screenshot: str
    requestId: Optional[str] = None


@app.post('/analyze')
async def analyze_screenshot(request: ScreenshotRequest):
    try:
        request_id = request.requestId or f'auto_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{os.urandom(4).hex()}'
        logger.info(f'Processing analysis request: {request_id}')

        base64_data = request.screenshot.split(',')[1] if ',' in request.screenshot else request.screenshot
        image_data = base64.b64decode(base64_data)
        pil_image = Image.open(io.BytesIO(image_data))

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        analysis_table.insert([{'image': pil_image, 'timestamp': datetime.now(), 'request_id': request_id}])

        result = (
            analysis_table.where(analysis_table.request_id == request_id)
            .select(
                analysis_table.levels,
                analysis_table.indicators,
                analysis_table.trade_setup,
                analysis_table.signal_type,
                analysis_table.summary,
            )
            .tail(1)
        )

        if not result:
            raise HTTPException(status_code=500, detail='No analysis results found')

        return {
            'request_id': request_id,
            'support_levels': result['levels'][0]['support'],
            'resistance_levels': result['levels'][0]['resistance'],
            'technical_indicators': result['indicators'][0],
            'trade_setup': result['trade_setup'][0],
            'signal_type': result['signal_type'][0],
            'summary': result['summary'][0],
        }

    except Exception as e:
        logger.error(f'Analysis error for request {request_id if "request_id" in locals() else "unknown"}: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    logger.info('Starting server...')
    uvicorn.run(app, host='0.0.0.0', port=8000)
