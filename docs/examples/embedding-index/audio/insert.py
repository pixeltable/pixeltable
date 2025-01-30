import pixeltable as pxt

audio_t = pxt.get_table('audio_index.audio')

audio_t.insert([{'audio_file': 's3://pixeltable-public/audio/10-minute tour of Pixeltable.mp3'}])
