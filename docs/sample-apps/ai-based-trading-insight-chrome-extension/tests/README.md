# Trading Analysis API Tests

Simple test suite for the Trading Analysis API that processes trading charts and provides technical analysis.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install pytest pillow numpy requests fastapi uvicorn python-dotenv

# Create .env file
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

## Running Tests

1. Start the FastAPI server:
```bash
python main.py
```

2. In another terminal, run tests:
```bash
pytest test.py -v
```

## Test Cases

- Server connection
- Response format
- Support/resistance levels
- Technical indicators
- Trade setup
- Signal type
- Error handling

## Using Custom Test Images

1. Put your test chart images in a `test_data` folder
2. Update the `sample_image` fixture in `test_trading_analysis.py`

## Troubleshooting

- Make sure FastAPI server is running on port 8000
- Verify your ANTHROPIC_API_KEY is valid
- Check if test images are valid JPG/PNG files