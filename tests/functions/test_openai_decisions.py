import asyncio
import json
from typing import Any, Iterator

import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable.utils.image import to_base64

from ..utils import SAMPLE_IMAGE_FILE_PATH, skip_test_if_not_installed, validate_update_status

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')

QUESTIONS: list[dict[str, Any]] = [
    {'type': 'predicate', 'name': 'complaint', 'instructions': 'Is the review a complaint?'},
    {
        'type': 'choice',
        'name': 'topic',
        'instructions': 'What is the review about?',
        'choices': [{'value': 'shipping'}, {'value': True}],
    },
    {
        'type': 'score',
        'name': 'sentiment',
        'instructions': 'How positive is the review?',
        'levels': [{'label': 'negative'}, {'label': 'neutral'}, {'label': 'positive'}],
    },
    {'type': 'predicate', 'name': 'secret', 'instructions': 'REFUSE'},
    {'type': 'predicate', 'instructions': 'Is the review in English?'},
]


class MockOpenaiApi:
    """Serves synthetic Decisions and Responses replies and records each request body."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.client_calls = 0
        self.delay = 0.0
        self.in_flight = 0
        self.max_in_flight = 0

    async def handle(self, request: Any) -> Any:
        import httpx2

        await request.aread()
        body = json.loads(request.content)
        self.requests.append(body)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        if request.url.path.endswith('/responses'):
            usage = {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}
            return httpx2.Response(200, json={'id': 'resp_1', 'object': 'response', 'output': [], 'usage': usage})
        assert request.url.path.endswith('/decisions')
        return httpx2.Response(200, json=self.decision(body))

    @classmethod
    def decision(cls, body: dict[str, Any]) -> dict[str, Any]:
        return {
            'answers': [cls.answer(q) for q in body['questions']],
            'model': body['model'],
            'usage': {
                'input_tokens': 10,
                'input_tokens_details': {'cache_write_tokens': 0, 'cached_tokens': 0},
                'output_tokens': 2,
                'output_tokens_details': {'reasoning_tokens': 0},
                'total_tokens': 12,
            },
        }

    @classmethod
    def answer(cls, question: dict[str, Any]) -> dict[str, Any]:
        # an unnamed question gets an answer without a `name` key
        answer: dict[str, Any] = {'name': question['name']} if 'name' in question else {}
        if question['instructions'] == 'REFUSE':
            return {**answer, 'type': 'refusal'}
        answer['type'] = question['type']
        if question['type'] == 'predicate':
            answer['probability'] = 0.75
        elif question['type'] == 'choice':
            values = [c['value'] for c in question['choices']]
            answer['choice'] = values[-1]
            answer['confidence'] = 0.8
            answer['probabilities'] = [{'value': v, 'probability': 1 / len(values)} for v in values]
        else:
            labels = [level['label'] for level in question['levels']]
            answer['score'] = 2.0
            answer['confidence'] = 0.6
            answer['probabilities'] = [
                {'label': label, 'value': i, 'probability': 1 / len(labels)} for i, label in enumerate(labels)
            ]
        return answer


@pytest.fixture
def mock_api(init_env: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[MockOpenaiApi]:
    skip_test_if_not_installed('openai')
    import httpx2
    import openai

    from pixeltable.functions import openai as pxt_openai
    from pixeltable.runtime import get_runtime

    api = MockOpenaiApi()
    client = openai.AsyncOpenAI(
        api_key='test-key', max_retries=0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(api.handle))
    )

    def openai_client() -> openai.AsyncOpenAI:
        api.client_calls += 1
        return client

    monkeypatch.setattr(pxt_openai, '_openai_client', openai_client)
    yield api
    get_runtime().event_loop.run_until_complete(client.close())


def png_data_url(path: str) -> str:
    with PIL.Image.open(path) as img:
        return f'data:image/png;base64,{to_base64(img, format="png")}'


class TestOpenaiDecisions:
    def test_text_input(self, uses_db: None, mock_api: MockOpenaiApi) -> None:
        from pixeltable.functions.openai import decisions

        t = pxt.create_table('test_tbl', {'review': pxt.String | None})
        t.add_computed_column(
            decision=decisions(
                t.review, questions=QUESTIONS, model_kwargs={'prompt_cache_key': 'reviews', 'safety_identifier': 'u1'}
            )
        )
        validate_update_status(t.insert([{'review': 'Arrived late.'}, {'review': None}]), expected_rows=2)

        assert len(mock_api.requests) == 1
        body = mock_api.requests[0]
        assert body['input'] == 'Arrived late.'
        assert body['model'] == 'gpt-6-luna'
        assert body['questions'] == QUESTIONS
        assert body['prompt_cache_key'] == 'reviews'
        assert body['safety_identifier'] == 'u1'

        answers = t.decision.answers
        projections = {
            'complaint': answers[0].probability,
            'topic': answers[1].choice,
            'topic_confidence': answers[1].confidence,
            'topic_values': answers[1].probabilities['*'].value,
            'sentiment': answers[2].score,
            'sentiment_labels': answers[2].probabilities['*'].label,
            'refusal_type': answers[3].type,
            'refusal_probability': answers[3].probability,
            'unnamed': answers[4].name,
            'model': t.decision.model,
            'total_tokens': t.decision.usage.total_tokens,
        }
        for name in ('complaint', 'sentiment', 'refusal_probability'):
            col_type = projections[name].col_type
            assert col_type.is_float_type() and col_type.nullable, name
        rows = t.order_by(t.review).select(**projections).collect()
        assert rows[0] == {
            'complaint': 0.75,
            'topic': True,
            'topic_confidence': 0.8,
            'topic_values': ['shipping', True],
            'sentiment': 2.0,
            'sentiment_labels': ['negative', 'neutral', 'positive'],
            'refusal_type': 'refusal',
            'refusal_probability': None,
            'unnamed': None,
            'model': 'gpt-6-luna',
            'total_tokens': 12,
        }
        # a null input short-circuits to a null result without a request
        assert all(v is None for v in rows[1].values())

    def test_message_input(self, uses_db: None, mock_api: MockOpenaiApi) -> None:
        from pixeltable.functions.openai import decisions

        public_url = 'https://example.com/photo.png'
        data_url = 'data:image/png;base64,iVBORw0KGgo='
        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': 'Product photos:'},
                    {'type': 'input_image', 'image_url': t.image},
                    {'type': 'input_image', 'image_url': public_url},
                    {'type': 'input_image', 'image_url': data_url, 'detail': 'low'},
                ],
            }
        ]
        t.add_computed_column(decision=decisions(messages, questions=QUESTIONS[:1]))
        validate_update_status(t.insert([{'image': SAMPLE_IMAGE_FILE_PATH}]), expected_rows=1)

        assert mock_api.requests[0]['input'] == [
            {
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': 'Product photos:'},
                    {'type': 'input_image', 'image_url': png_data_url(SAMPLE_IMAGE_FILE_PATH)},
                    {'type': 'input_image', 'image_url': public_url},
                    {'type': 'input_image', 'image_url': data_url, 'detail': 'low'},
                ],
            }
        ]
        assert t.select(p=t.decision.answers[0].probability).collect()[0]['p'] == pytest.approx(0.75)

    def test_serialize_input_images_copies_input(self, init_env: None) -> None:
        """Call the helper directly because UDF argument evaluation hides mutations of the caller's objects."""
        from pixeltable.functions.openai import _serialize_input_images

        with PIL.Image.open(SAMPLE_IMAGE_FILE_PATH) as img:
            img.load()
        part = {'type': 'input_image', 'image_url': img}
        original = [{'role': 'user', 'content': [part]}, 'other']
        serialized = _serialize_input_images(original)
        assert serialized[0]['content'][0]['image_url'] == png_data_url(SAMPLE_IMAGE_FILE_PATH)
        assert serialized[1] == 'other'
        assert part['image_url'] is img

    @pytest.mark.parametrize('question', QUESTIONS[:3])
    def test_question_instruction_images(
        self, uses_db: None, mock_api: MockOpenaiApi, question: dict[str, Any]
    ) -> None:
        from pixeltable.functions.openai import decisions

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        instructions = [
            {'type': 'input_text', 'text': 'Compare the product photos.'},
            {'type': 'input_image', 'image_url': t.image},
            {'type': 'input_image', 'image_url': 'https://example.com/photo.png'},
            {'type': 'input_image', 'image_url': 'data:image/png;base64,iVBORw0KGgo=', 'detail': 'low'},
        ]
        t.add_computed_column(
            decision=decisions('Product photos', questions=[{**question, 'instructions': instructions}])
        )
        validate_update_status(t.insert([{'image': SAMPLE_IMAGE_FILE_PATH}]), expected_rows=1)

        assert mock_api.requests[0]['questions'] == [
            {
                **question,
                'instructions': [
                    instructions[0],
                    {'type': 'input_image', 'image_url': png_data_url(SAMPLE_IMAGE_FILE_PATH)},
                    instructions[2],
                    instructions[3],
                ],
            }
        ]
        assert t.select(answer=t.decision.answers[0].type).collect()[0]['answer'] == question['type']

    @pytest.mark.parametrize('tool_type', ['function', 'custom_tool'])
    def test_tool_output_images(self, uses_db: None, mock_api: MockOpenaiApi, tool_type: str) -> None:
        from pixeltable.functions.openai import decisions

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        call = {'type': f'{tool_type}_call', 'call_id': 'call_1', 'name': 'product_photo'}
        call['arguments' if tool_type == 'function' else 'input'] = '{}'
        output = [
            {'type': 'input_image', 'image_url': t.image},
            {'type': 'input_image', 'image_url': 'https://example.com/photo.png'},
            {'type': 'input_image', 'image_url': 'data:image/png;base64,iVBORw0KGgo='},
        ]
        messages = [call, {'type': f'{tool_type}_call_output', 'call_id': 'call_1', 'output': output}]
        t.add_computed_column(decision=decisions(messages, questions=QUESTIONS[:1]))
        validate_update_status(t.insert([{'image': SAMPLE_IMAGE_FILE_PATH}]), expected_rows=1)

        assert mock_api.requests[0]['input'] == [
            call,
            {
                'type': f'{tool_type}_call_output',
                'call_id': 'call_1',
                'output': [
                    {'type': 'input_image', 'image_url': png_data_url(SAMPLE_IMAGE_FILE_PATH)},
                    output[1],
                    output[2],
                ],
            },
        ]
        assert t.select(p=t.decision.answers[0].probability).collect()[0]['p'] == pytest.approx(0.75)

    def test_responses_image_serialization(self, uses_db: None, mock_api: MockOpenaiApi) -> None:
        from pixeltable.functions.openai import responses

        t = pxt.create_table('test_tbl', {'image': pxt.Image})
        messages = [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': t.image}]}]
        t.add_computed_column(response=responses(messages, model='gpt-4o-mini'))
        validate_update_status(t.insert([{'image': SAMPLE_IMAGE_FILE_PATH}]), expected_rows=1)

        image_part = mock_api.requests[0]['input'][0]['content'][0]
        assert image_part == {'type': 'input_image', 'image_url': png_data_url(SAMPLE_IMAGE_FILE_PATH)}

    def test_question_count(self, uses_db: None, mock_api: MockOpenaiApi) -> None:
        from pixeltable.functions.openai import decisions

        question = QUESTIONS[0]
        t = pxt.create_table('test_tbl', {'review': pxt.String})
        for n in (1, 400):
            t.add_computed_column(**{f'd_{n}': decisions(t.review, questions=[question] * n)})
        validate_update_status(t.insert([{'review': 'Fine.'}]), expected_rows=1)
        assert sorted(len(body['questions']) for body in mock_api.requests) == [1, 400]
        row = t.select(n1=t.d_1.answers, n400=t.d_400.answers).collect()[0]
        assert len(row['n1']) == 1 and len(row['n400']) == 400

        mock_api.requests.clear()
        t.add_computed_column(d_empty=decisions(t.review, questions=[]), on_error='ignore')
        msg = t.select(msg=t.d_empty.errormsg).collect()[0]['msg']
        assert '`questions` must contain at least one question' in msg
        assert mock_api.requests == []

    def test_invalid_input(self, uses_db: None, mock_api: MockOpenaiApi) -> None:
        from pixeltable.functions.openai import decisions

        t = pxt.create_table('test_tbl', {'input': pxt.Json})
        t.add_computed_column(decision=decisions(t.input, questions=QUESTIONS[:1]))
        status = t.insert([{'input': 'first'}, {'input': 5}, {'input': 'last'}], on_error='ignore')
        assert status.num_excs == 1
        rows = t.select(t.input, msg=t.decision.errormsg, p=t.decision.answers[0].probability).collect()
        by_input = {json.dumps(row['input']): row for row in rows}
        assert '`input` must be a string or a list of input items, got int' in by_input['5']['msg']
        assert by_input['"first"']['p'] == by_input['"last"']['p'] == pytest.approx(0.75)
        assert sorted(body['input'] for body in mock_api.requests) == ['first', 'last']

    def test_sdk_version(self, uses_db: None, mock_api: MockOpenaiApi, monkeypatch: pytest.MonkeyPatch) -> None:
        from pixeltable import env
        from pixeltable.functions.openai import decisions

        def require_package(self: Any, package_name: str, min_version: list[int] | None = None) -> None:
            if min_version == [3, 32]:
                raise pxt.RequestError(pxt.ErrorCode.UNSUPPORTED_OPERATION, 'openai>=3.32 is required')

        monkeypatch.setattr(env.Env, 'require_package', require_package)
        t = pxt.create_table('test_tbl', {'review': pxt.String})
        t.add_computed_column(decision=decisions(t.review, questions=QUESTIONS[:1]))
        status = t.insert([{'review': 'Fine.'}], on_error='ignore')
        assert status.num_excs == 1
        assert 'openai>=3.32 is required' in t.select(msg=t.decision.errormsg).collect()[0]['msg']
        assert mock_api.client_calls == 0
        assert mock_api.requests == []

    def test_concurrent_requests(self, uses_db: None, mock_api: MockOpenaiApi) -> None:
        from pixeltable.functions.openai import decisions

        mock_api.delay = 0.5
        t = pxt.create_table('test_tbl', {'review': pxt.String})
        t.add_computed_column(decision=decisions(t.review, questions=QUESTIONS[:1]))
        num_rows = 20
        validate_update_status(t.insert({'review': f'review {i}'} for i in range(num_rows)), expected_rows=num_rows)
        assert len(mock_api.requests) == num_rows
        assert mock_api.max_in_flight > 1
        assert [row['p'] for row in t.select(p=t.decision.answers[0].probability).collect()] == [0.75] * num_rows
