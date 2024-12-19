import logging

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

ALLOWED_TYPES = {
    'document': [
        'application/pdf',
        'text/markdown',
        'text/html',
        'text/plain',
        'application/xml', 'text/xml',
        'application/x-pdf',
        'text/x-markdown',
        'text/x-html',
        'text/x-xml'
    ],
    'video': [
        'video/mp4',
        'video/quicktime',
        'video/x-msvideo',
        'video/webm'
    ],
    'audio': [
        'audio/mpeg',
        'audio/wav',
        'audio/ogg',
        'audio/webm'
    ]
}
