"""Trading insight schema — idempotent by default.

python schema.py
RESET_SCHEMA=true python schema.py
"""

import os

import config
import functions

import pixeltable as pxt
from pixeltable.functions import image as pxt_image
from pixeltable.functions.anthropic import messages

if os.getenv('RESET_SCHEMA', 'false').lower() == 'true':
    pxt.drop_dir(config.APP_NAMESPACE, force=True)

pxt.create_dir(config.APP_NAMESPACE, if_exists='ignore')

analysis_table = pxt.create_table(
    f'{config.APP_NAMESPACE}.analysis',
    {'image': pxt.Image, 'timestamp': pxt.Timestamp, 'request_id': pxt.String},
    if_exists='ignore',
)

analysis_table.add_computed_column(
    image_raw=pxt_image.b64_encode(analysis_table.image, 'jpeg'),
    if_exists='ignore',
)
analysis_table.add_computed_column(
    msg_list=functions.create_messages(analysis_table.image_raw),
    if_exists='ignore',
)
analysis_table.add_computed_column(
    claude_response=messages(
        model=config.LLM_MODEL,
        messages=analysis_table.msg_list,
        max_tokens=1500,
        model_kwargs={
            'system': config.SYSTEM_PROMPT,
            'temperature': 0.1,
            'top_p': 0.3,
            'top_k': 10,
        },
    ),
    if_exists='ignore',
)
analysis_table.add_computed_column(
    analysis_text=analysis_table.claude_response.content[0].text,
    if_exists='ignore',
)
analysis_table.add_computed_column(
    levels=functions.parse_support_resistance(analysis_table.analysis_text),
    if_exists='ignore',
)
analysis_table.add_computed_column(
    indicators=functions.extract_technical_indicators(analysis_table.analysis_text),
    if_exists='ignore',
)
analysis_table.add_computed_column(
    trade_setup=functions.extract_trade_setup(analysis_table.analysis_text),
    if_exists='ignore',
)
analysis_table.add_computed_column(
    signal_type=functions.extract_signal_type(analysis_table.analysis_text),
    if_exists='ignore',
)
analysis_table.add_computed_column(
    summary=functions.parse_summary(analysis_table.analysis_text),
    if_exists='ignore',
)

if __name__ == '__main__':
    print('Schema setup complete.')
