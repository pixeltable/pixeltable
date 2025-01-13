from dotenv import load_dotenv
import os

load_dotenv()

def check_api_key(key_name: str) -> str:
    """Validate presence of API key in environment.

    Args:
        key_name: Name of the API key to check

    Returns:
        str: The API key value

    Raises:
        ValueError: If API key is not found
    """
    if api_key := os.getenv(key_name):
        return api_key
    raise ValueError(f"API key {key_name} not found in .env file")