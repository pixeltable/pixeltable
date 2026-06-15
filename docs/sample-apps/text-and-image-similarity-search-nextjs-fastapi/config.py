import os

APP_NAMESPACE = 'media_search'
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
CLIP_MODEL_ID = 'openai/clip-vit-base-patch32'
