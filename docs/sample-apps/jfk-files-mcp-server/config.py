"""
Configuration settings for the JFK Files MCP Server application.

This file centralizes the application configuration parameters.
"""

# The directory name used for storing the JFK files data within Pixeltable.
# This determines where Pixeltable will create the database table structures.
DIRECTORY = 'JFK'

# The specific Mistral AI model used for OCR text extraction and document summarization.
# 'mistral-small-latest' provides a good balance between performance and accuracy.
MISTRAL_MODEL = 'mistral-small-latest'
