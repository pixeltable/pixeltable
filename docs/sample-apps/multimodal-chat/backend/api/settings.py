import logging

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

ALLOWED_TYPES = {
    "document": [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/x-python",
    ],
    "video": ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"],
    "audio": ["audio/mpeg", "audio/wav", "audio/ogg", "audio/webm"],
}
