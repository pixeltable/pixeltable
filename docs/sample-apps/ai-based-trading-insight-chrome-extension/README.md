# AI-based Day Trading Insights - Chrome Extension 📈

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/) [![Pixeltable](https://img.shields.io/badge/powered%20by-Pixeltable-purple)](https://github.com/pixeltable/pixeltable) [![Anthropic](https://img.shields.io/badge/AI-Claude%203%20Sonnet-blue)](https://www.anthropic.com/)

A Chrome extension powered by Pixeltable and Claude Sonnet that provides real-time technical analysis and trading insights for stock market charts. It leverages advanced AI to analyze chart patterns, technical indicators, and market conditions to help traders make more informed decisions.

<img src="store-assets\presentation.png" width="50%" alt="presentation"/>

## 🌟 Features

### Real-time Technical Analysis

Analyze the captured image to detect:
- Support & Resistance Level Detection
- Key Technical Indicators (MACD, RSI, Stochastic)
- Volume Analysis & Price Action
- VWAP and more

### Smart Trade Setup
- Automated Entry Point Detection
- Dynamic Stop Loss Calculation
- Price Target Generation

### AI-Powered Pattern Recognition
- Chart Pattern Identification
- Trend Analysis & Direction
- Signal Generation (Bullish/Bearish/Neutral)
- Multi-timeframe Confirmation if available

## 🔄 Request Flow & ID Tracking

The extension uses a request ID system to ensure accurate matching between requests and responses:

### Request ID System
1. **Generation & Format**
   ```typescript
   // Client-side generation (Chrome extension)
   function generateRequestId(): string {
     const timestamp = Date.now();
     const random = Math.random().toString(36).substring(2, 10);
     return `${timestamp}_${random}`;
   }
   ```

2. **ID Implementation**
   ```python
   class ScreenshotRequest(BaseModel):
       screenshot: str
       requestId: Optional[str] = None

       @property
       def get_request_id(self) -> str:
           if not self.requestId:
               timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
               self.requestId = f"auto_{timestamp}_{os.urandom(4).hex()}"
           return self.requestId
   ```

### Request-Response Flow
1. **Client Generation**: Extension creates unique ID per request
2. **Server Processing**: Backend tracks analysis with ID
3. **Response Matching**: Results paired via ID verification
4. **Error Tracking**: All logs tagged with request IDs

## 🚀 Installation

### Backend Setup
```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend
python -m uvicorn main:app
```

### Extension Setup
1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `dist` directory to upload all the related files

## 🔧 Technology Stack

### Backend Infrastructure
- **FastAPI**: Backend framework
- **Pixeltable**: AI data infrastructure
- **Anthropic**: LLM

### Extension Components
- Chrome Extension API
- HTML, CSS, & JavaScript

## 📊 System Architecture

### Data Flow
1. **Chart Capture**: Extension captures current view
2. **Image Processing**: Pixeltable stores image, build custom function, and orchestrate data transformation and API calls.
3. **AI Analysis**: Claude processes chart
4. **Data Processing**: Results structured via Pixeltable
5. **Response Delivery**: Analysis displayed in UI

```mermaid
sequenceDiagram
    participant C as Chrome Extension
    participant B as FastAPI (Server)
    participant P as Pixeltable
    participant Cl as Claude Sonnet
    participant DB as Pixeltable

    rect rgb(240, 248, 255)
        Note over C,DB: Phase 1: Image Capture & Initial Processing
        C->>C: Capture chart screenshot
        C->>C: Generate unique requestId
        C->>B: POST /analyze with base64 image
        B->>P: Initialize analysis table
    end

    rect rgb(255, 245, 245)
        Note over B,Cl: Phase 2: Image Analysis & AI Processing
        P->>P: Process image through computed columns
        P->>Cl: Send image for analysis with prompt
        Cl-->>P: Return structured analysis
        P->>P: Parse analysis into components
    end

    rect rgb(245, 255, 245)
        Note over P,DB: Phase 3: Data Processing & Storage
        P->>P: Extract technical indicators
        P->>P: Parse support/resistance levels
        P->>P: Determine signal type
        P->>P: Generate trade setup
        P->>DB: Store analysis results
    end

    rect rgb(245, 245, 255)
        Note over B,C: Phase 4: Response Delivery
        B->>P: Query results by requestId
        P-->>B: Return processed analysis
        B-->>C: Send structured JSON response
        C->>C: Update UI with results
    end

    rect rgb(255, 250, 240)
        Note over C,DB: Phase 5: Error Handling
        alt Error occurs
            B->>B: Log error with requestId
            B-->>C: Return error status
            C->>C: Display error message
        end
    end
```

## 🔍 Troubleshooting

### Common Problems
- Extension not loading: Check manifest.json
- Analysis timeout: Verify API keys
- Image processing: Check screenshot format

## ⚠️ Disclaimer

This application is for educational and research purposes only. Do not use it to make financial decisions. Always consult with a licensed financial advisor before making any investment decisions.