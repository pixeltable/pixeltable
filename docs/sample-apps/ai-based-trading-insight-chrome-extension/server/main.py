import base64
import io
import logging
import os
from datetime import datetime

import config
import schema  # noqa: F401 — initializes Pixeltable schema on import
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

import pixeltable as pxt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

if not config.ANTHROPIC_API_KEY:
    raise RuntimeError('ANTHROPIC_API_KEY is required. Copy server/.env.example to server/.env and set your key.')

os.environ['ANTHROPIC_API_KEY'] = config.ANTHROPIC_API_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*']
)

analysis_table = pxt.get_table(f'{config.APP_NAMESPACE}.analysis')


class ScreenshotRequest(BaseModel):
    screenshot: str
    requestId: str | None = None


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
        logger.error(f'Analysis error: {e!s}')
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == '__main__':
    import uvicorn

    logger.info('Starting server...')
    uvicorn.run(app, host='0.0.0.0', port=8000)
