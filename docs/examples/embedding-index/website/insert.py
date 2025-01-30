import pixeltable as pxt

website_index = pxt.get_table('website_index.websites')

# Website ingestion pipeline (read, parse, and store)
website_index.insert([{"website": "https://quotes.toscrape.com/"}])
