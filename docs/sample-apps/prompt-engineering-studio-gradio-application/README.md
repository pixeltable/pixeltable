# Prompt Engineering Studio with Pixeltable and Gradio

[![Open In HuggingFace](https://img.shields.io/badge/Live-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/Pixeltable/Prompt-Engineering-and-LLM-Studio)

## Overview

This project demonstrates how to build a prompt engineering and LLM testing application using Pixeltable for data storage, transformation, and computation and Gradio for the user interface. The application helps with studying different prompts, compare model responses, and track results over time.

## Features

- 🔄 **Test and Compare Models**: Compare responses from different LLM models side by side
- 📊 **Advanced Analytics**: Automatic analysis of model outputs including custom metrics
- 📝 **Experiment Tracking**: Automatic versioning and history of all prompts and results
- 🎛️ **Parameter Tuning**: Control temperature, top_p, and other model parameters
- 📈 **Results Comparison**: Query model outputs and analytics
- 🔍 **Historical Analysis**: Review past experiments

## Getting Started

### Prerequisites

```bash
uv sync
jupyter notebook pixeltable-and-gradio-application.ipynb
```

You'll also need:
- A Mistral AI API key
- Python 3.10 or later

### Running the Application

1. Clone this repository:
```bash
git clone https://github.com/pixeltable/pixeltable.git
cd pixeltable/docs/sample-apps/prompt-engineering-studio-gradio-application
```

2. Set your API key:
```bash
export MISTRAL_API_KEY='your-api-key-here'
```

3. Run the application:
```bash
jupyter notebook pixeltable-and-gradio-application.ipynb
```

## How It Works

The application combines several key technologies:

- **Pixeltable**: Handles data management, versioning, and computed columns
- **Gradio**: Provides the web interface and interactive elements
- **TextBlob & NLTK**: Powers the text analysis features
- **Mistral AI**: Provides the LLM capabilities

### Key Components

1. **Data Management**
   - Automatic storage of all prompts and responses
   - Version tracking of experiments
   - Computed columns for analytics

2. **Analysis Pipeline**
   - Sentiment analysis of responses
   - Readability scoring
   - Keyword extraction
   - Response comparison

3. **User Interface**
   - Input fields for prompts and parameters
   - Real-time response display
   - Tabbed interface for different views
   - Historical experiment tracking