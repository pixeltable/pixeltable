import pixeltable as pxt


@pxt.udf
def create_prompt(doc_context: list[dict], video_context: list[dict], audio_context: list[dict], question: str) -> str:
    context_parts = []
    if doc_context:
        context_parts.append(
            'Document Context:\n' + '\n\n'.join(item['text'] for item in doc_context if item and 'text' in item)
        )
    if video_context:
        context_parts.append(
            'Video Context:\n' + '\n\n'.join(item['text'] for item in video_context if item and 'text' in item)
        )
    if audio_context:
        context_parts.append(
            'Audio Context:\n' + '\n\n'.join(item['text'] for item in audio_context if item and 'text' in item)
        )
    full_context = '\n\n---\n\n'.join(context_parts) if context_parts else 'No relevant context found.'
    return f'Context:\n{full_context}\n\nQuestion:\n{question}'


@pxt.udf
def create_messages(history: list[dict], prompt: str) -> list[dict]:
    messages = [
        {
            'role': 'system',
            'content': 'You are a helpful AI assistant maintaining conversation context while answering questions based on provided sources.',
        }
    ]
    messages.extend({'role': msg['role'], 'content': msg['content']} for msg in history)
    messages.append({'role': 'user', 'content': prompt})
    return messages
