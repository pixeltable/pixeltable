import os

import pixeltable as pxt
from pixeltable.functions.huggingface import clip

DIRECTORY = 'image_index'
TABLE_NAME = f'{DIRECTORY}.image'
VIEW_NAME = f'{DIRECTORY}.image_chunks'
RECREATE = True

if RECREATE:
    pxt.drop_dir(DIRECTORY, force=True)

if TABLE_NAME not in pxt.list_tables():
    # Create image table
    pxt.create_dir(DIRECTORY, if_exists='ignore')
    image_index = pxt.create_table(TABLE_NAME, {'image_file': pxt.Image}, if_exists='ignore')

    # Create view that chunks text into sentences
    image_view = pxt.create_view(VIEW_NAME, image_index, if_exists='ignore')

    # Define the embedding model
    embed_model = clip.using(model_id='openai/clip-vit-base-patch32')

    # Create embedding index
    image_view.add_embedding_index(column='image_file', image_embed=embed_model, if_exists='ignore')

else:
    image_index = pxt.get_table(TABLE_NAME)
    image_view = pxt.get_table(VIEW_NAME)

# Add data to the table
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
image_folder = os.path.join(WORKSPACE_ROOT, 'docs', 'resources', 'images')
image_index.insert([{'image_file': os.path.join(image_folder, image_path)} for image_path in os.listdir(image_folder)])

# Load in a reference image
import PIL

reference_image = PIL.Image.open(os.path.join(image_folder, '000000000001.jpg'))

# Perform a similarity search
sim = image_view.image.similarity(reference_image)

# image_index.select(image_index.image_file, sim).order_by(sim, asc=False).limit(5)
