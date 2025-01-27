import base64
import pytest
import requests
from pathlib import Path
from typing import Optional

class TestConfig:
    BASE_URL = "http://localhost:8000"
    TEST_REQUEST_ID = "test_request_001"

# Validation constants
REQUIRED_TECHNICAL_INDICATORS = {'macd', 'rsi', 'mfi', 'stochastic', 'volume', 'current_price', 'vwap'}
REQUIRED_TRADE_SETUP_FIELDS = {'entry', 'stop_loss', 'target'}
VALID_SIGNAL_TYPES = {'BULLISH', 'BEARISH', 'NEUTRAL'}

@pytest.fixture
def base_url():
    return TestConfig.BASE_URL

@pytest.fixture
def sample_image() -> Path:
    """Use actual test chart image from test_data directory."""
    image_path = Path("test_data/test-chart.png")
    if not image_path.exists():
        raise FileNotFoundError(
            "Test chart image not found. Please place test_chart.png in test-data directory"
        )
    return image_path

def encode_image_to_base64(image_path: Path) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@pytest.fixture
def analysis_payload(sample_image):
    """Create a payload for the analysis endpoint."""
    base64_image = encode_image_to_base64(sample_image)
    return {
        "screenshot": f"data:image/jpeg;base64,{base64_image}",
        "requestId": TestConfig.TEST_REQUEST_ID
    }

def validate_levels(levels: list) -> bool:
    """Validate support/resistance levels format."""
    if not isinstance(levels, list) or len(levels) != 3:
        return False
    return all(level is None or isinstance(level, float) for level in levels)

def validate_indicators(indicators: dict) -> bool:
    """Validate technical indicators format."""
    return all(key in indicators for key in REQUIRED_TECHNICAL_INDICATORS)

def validate_trade_setup(setup: dict) -> bool:
    """Validate trade setup format."""
    return all(key in setup for key in REQUIRED_TRADE_SETUP_FIELDS)

class TestTradingAnalysis:
    
    def test_analyze_endpoint_connection(self, base_url):
        """Test if the server is running and accessible."""
        try:
            response = requests.get(base_url)
            assert response.status_code in [200, 404]  # 404 is ok as root path might not be defined
        except requests.ConnectionError:
            pytest.fail("Server is not running")

    def test_analyze_endpoint_response_structure(self, base_url, analysis_payload):
        """Test if the analyze endpoint returns the expected response structure."""
        response = requests.post(
            f"{base_url}/analyze",
            json=analysis_payload
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)
        
        # Check all required fields are present
        required_fields = {
            'request_id',
            'support_levels',
            'resistance_levels',
            'technical_indicators',
            'trade_setup',
            'signal_type',
            'summary'
        }
        assert all(field in data for field in required_fields)

    def test_support_resistance_levels(self, base_url, analysis_payload):
        """Test support and resistance levels format."""
        response = requests.post(
            f"{base_url}/analyze",
            json=analysis_payload
        )
        
        data = response.json()
        assert validate_levels(data['support_levels'])
        assert validate_levels(data['resistance_levels'])

    def test_technical_indicators(self, base_url, analysis_payload):
        """Test technical indicators format."""
        response = requests.post(
            f"{base_url}/analyze",
            json=analysis_payload
        )
        
        data = response.json()
        assert validate_indicators(data['technical_indicators'])

    def test_trade_setup(self, base_url, analysis_payload):
        """Test trade setup format."""
        response = requests.post(
            f"{base_url}/analyze",
            json=analysis_payload
        )
        
        data = response.json()
        assert validate_trade_setup(data['trade_setup'])

    def test_signal_type(self, base_url, analysis_payload):
        """Test signal type validation."""
        response = requests.post(
            f"{base_url}/analyze",
            json=analysis_payload
        )
        
        data = response.json()
        assert data['signal_type'] in VALID_SIGNAL_TYPES

    @pytest.mark.parametrize("bad_payload,expected_status", [
        ({}, 422),  # Empty payload - Validation error
        ({"screenshot": "invalid_base64"}, 500),  # Invalid base64 - Processing error
        ({"screenshot": "data:image/jpeg;base64,SGVsbG8="}, 500)  # Invalid image data - Processing error
    ])
    def test_analyze_endpoint_error_handling(self, base_url, bad_payload, expected_status):
        """Test error handling with invalid payloads."""
        response = requests.post(
            f"{base_url}/analyze",
            json=bad_payload
        )
        assert response.status_code == expected_status

    def test_request_id_persistence(self, base_url, analysis_payload):
        """Test if the request ID is preserved in the response."""
        response = requests.post(
            f"{base_url}/analyze",
            json=analysis_payload
        )
        
        data = response.json()
        assert data['request_id'] == TestConfig.TEST_REQUEST_ID