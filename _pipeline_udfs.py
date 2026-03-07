"""
Complex pipeline UDFs — vision, NLP, and agent chains.
Must live in a named module (not __main__) for Pixeltable >= 0.5.20.
"""

import pixeltable as pxt
import json
import hashlib
import numpy as np


# ── Embedding UDF (lightweight mock for indexing) ────────────────────────────


@pxt.udf
def mock_embed(text: str) -> pxt.Array[(8,), pxt.Float]:
    h = hashlib.sha256(text.encode()).digest()
    arr = np.frombuffer(h, dtype=np.float32).copy()
    norm = np.linalg.norm(arr)
    if norm < 1e-7:
        arr = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
    else:
        arr = arr / norm
    return np.nan_to_num(arr, nan=0.0).astype(np.float32)


# ── Vision UDFs ──────────────────────────────────────────────────────────────


@pxt.udf
def detect_objects(image_url: str) -> pxt.Json:
    h = int(hashlib.md5(image_url.encode()).hexdigest()[:8], 16)
    labels = ['person', 'car', 'dog', 'cat', 'bicycle', 'tree', 'building', 'chair', 'phone', 'laptop']
    n = (h % 5) + 1
    dets = []
    for i in range(n):
        obj = labels[(h + i) % len(labels)]
        conf = round(0.5 + (((h >> i) % 50) / 100), 2)
        x, y = (h + i * 37) % 100, (h + i * 53) % 100
        dets.append({'label': obj, 'confidence': conf, 'bbox': [x, y, x + 20, y + 20]})
    return dets


@pxt.udf
def count_detections(objects_json: pxt.Json) -> int:
    if isinstance(objects_json, str):
        objects_json = json.loads(objects_json)
    return len(objects_json)


@pxt.udf
def check_person(objects_json: pxt.Json) -> bool:
    if isinstance(objects_json, str):
        objects_json = json.loads(objects_json)
    return any(d['label'] == 'person' for d in objects_json)


@pxt.udf
def classify_scene(image_url: str, objects_json: pxt.Json) -> str:
    if isinstance(objects_json, str):
        objects_json = json.loads(objects_json)
    labels = {d['label'] for d in objects_json}
    if 'car' in labels and 'person' in labels:
        return 'street'
    if 'dog' in labels or 'cat' in labels:
        return 'domestic'
    if 'laptop' in labels or 'chair' in labels:
        return 'office'
    if 'building' in labels:
        return 'urban'
    return 'outdoor'


@pxt.udf
def assess_risk(scene_type: str, num_objects: int) -> float:
    base = {'street': 0.6, 'urban': 0.4, 'domestic': 0.1, 'office': 0.2, 'outdoor': 0.3}
    return round(min(base.get(scene_type, 0.3) + num_objects * 0.05, 1.0), 2)


@pxt.udf
def estimate_pose(image_url: str) -> str:
    h = int(hashlib.md5(image_url.encode()).hexdigest()[:4], 16)
    return ['standing', 'sitting', 'walking', 'running', 'crouching', 'waving'][h % 6]


@pxt.udf
def classify_action(pose_label: str, scene_type: str) -> str:
    if pose_label == 'running' and scene_type == 'street':
        return 'jogging'
    if pose_label == 'sitting' and scene_type == 'office':
        return 'working'
    if pose_label == 'waving':
        return 'greeting'
    if pose_label == 'crouching':
        return 'inspecting'
    return f'{pose_label}_in_{scene_type}'


# ── NLP UDFs ─────────────────────────────────────────────────────────────────


@pxt.udf
def detect_language(body: str) -> str:
    if any(c > '\u4e00' for c in body[:50]):
        return 'zh'
    if any('\u0600' < c < '\u06ff' for c in body[:50]):
        return 'ar'
    return 'en'


@pxt.udf
def analyze_sentiment(body: str) -> str:
    lower = body.lower()
    pos = sum(
        1 for w in ['great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'fantastic', 'powerful'] if w in lower
    )
    neg = sum(1 for w in ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'boring', 'fails'] if w in lower)
    return 'positive' if pos > neg else ('negative' if neg > pos else 'neutral')


@pxt.udf
def extract_entities(body: str) -> pxt.Json:
    entities = []
    for w in body.split():
        if len(w) > 2 and w[0].isupper() and w.isalpha():
            entities.append({'text': w, 'type': 'ORG' if len(w) > 6 else 'PERSON'})
    return entities[:10]


@pxt.udf
def summarize_text(body: str) -> str:
    sentences = body.replace('!', '.').replace('?', '.').split('.')
    summary = '. '.join(s.strip() for s in sentences[:2] if s.strip())
    return (summary + '.') if summary else body[:100]


@pxt.udf
def classify_topic(title: str, body: str) -> str:
    text = (title + ' ' + body).lower()
    topics = {
        'technology': [
            'ai',
            'machine learning',
            'data',
            'software',
            'algorithm',
            'model',
            'neural',
            'pixeltable',
            'compute',
        ],
        'science': ['research', 'experiment', 'hypothesis', 'study', 'discovery', 'physics', 'biology'],
        'business': ['revenue', 'market', 'startup', 'invest', 'growth', 'company', 'profit'],
        'health': ['medical', 'patient', 'treatment', 'health', 'clinical', 'disease'],
    }
    scores = {t: sum(1 for kw in kws if kw in text) for t, kws in topics.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'general'


@pxt.udf
def score_quality(body: str, sentiment: str, entities_json: pxt.Json) -> float:
    if isinstance(entities_json, str):
        entities_json = json.loads(entities_json)
    length_score = min(len(body.split()) / 100, 1.0)
    entity_score = min(len(entities_json) / 5, 1.0)
    sentiment_bonus = 0.1 if sentiment != 'neutral' else 0.0
    return round(length_score * 0.5 + entity_score * 0.3 + sentiment_bonus + 0.1, 2)


@pxt.udf
def compute_readability(body: str) -> float:
    words = body.split()
    sentences = max(body.count('.') + body.count('!') + body.count('?'), 1)
    avg_sent = len(words) / sentences
    avg_word = sum(len(w) for w in words) / max(len(words), 1)
    return round(max(0.0, min(100.0, 206.835 - 1.015 * avg_sent - 84.6 * (avg_word / 5))), 1)


@pxt.udf
def extract_phrases(summary: str, entities_json: pxt.Json) -> pxt.Json:
    if isinstance(entities_json, str):
        entities_json = json.loads(entities_json)
    entity_texts = [e['text'] for e in entities_json[:5]]
    summary_words = [w for w in summary.split() if len(w) > 4][:5]
    return list(set(entity_texts + summary_words))[:8]


# ── Agent UDFs ───────────────────────────────────────────────────────────────


@pxt.udf
def parse_intent(instruction: str) -> str:
    lower = instruction.lower()
    for intent, keywords in [
        ('search', ['search', 'find', 'look up', 'query']),
        ('create', ['create', 'generate', 'write', 'build']),
        ('analyze', ['analyze', 'evaluate', 'assess', 'examine']),
        ('update', ['update', 'modify', 'fix', 'change']),
        ('delete', ['delete', 'remove', 'drop', 'clean']),
    ]:
        if any(k in lower for k in keywords):
            return intent
    return 'unknown'


@pxt.udf
def assess_complexity(instruction: str, context: str) -> float:
    factors = 0.15
    if len(instruction.split()) > 20:
        factors += 0.2
    if len(context.split()) > 50:
        factors += 0.2
    if any(w in instruction.lower() for w in ['multiple', 'complex', 'all', 'every']):
        factors += 0.2
    if any(w in instruction.lower() for w in ['and', 'then', 'also', 'after']):
        factors += 0.15
    return round(min(factors, 1.0), 2)


@pxt.udf
def generate_plan(parsed_intent: str, context: str) -> str:
    plans = {
        'search': f'1. Parse query\n2. Search index\n3. Rank results\n4. Return from: {context[:40]}',
        'create': f'1. Validate input\n2. Generate template\n3. Fill context: {context[:40]}\n4. Finalize',
        'analyze': f'1. Collect data\n2. Apply framework\n3. Generate insights: {context[:40]}\n4. Summarize',
        'update': f'1. Locate resource\n2. Validate changes\n3. Apply to: {context[:40]}\n4. Verify',
        'delete': f'1. Confirm target\n2. Check deps\n3. Remove from: {context[:40]}\n4. Cleanup',
    }
    return plans.get(parsed_intent, f'1. Interpret\n2. Process: {context[:40]}\n3. Execute\n4. Report')


@pxt.udf
def chain_of_thought(plan: str, complexity: float) -> str:
    steps = plan.split('\n')
    reasoning = []
    for i, step in enumerate(steps):
        reasoning.append(f'Step {i + 1}: {step.strip()}')
        if complexity > 0.5 and i == 1:
            reasoning.append(f'  → Note: High complexity ({complexity}), need careful validation')
    reasoning.append(f'Conclusion: {len(steps)} steps, complexity={complexity}')
    return '\n'.join(reasoning)


@pxt.udf
def evaluate_confidence(reasoning: str) -> float:
    if len(reasoning) < 20:
        raise ValueError('Insufficient reasoning depth')
    base = 0.85
    if 'High complexity' in reasoning:
        base -= 0.2
    if len(reasoning.split('\n')) > 6:
        base -= 0.05
    if 'Note:' in reasoning:
        base -= 0.05
    return round(max(0.1, min(base, 1.0)), 2)


@pxt.udf
def make_decision(reasoning: str, confidence: float) -> str:
    if confidence >= 0.8:
        return 'approve'
    if confidence >= 0.5:
        return 'approve_with_review'
    if confidence >= 0.3:
        return 'defer'
    return 'reject'


@pxt.udf
def execute_action(decision: str, context: str) -> pxt.Json:
    if decision == 'reject':
        return {'status': 'rejected', 'reason': 'Low confidence'}
    if decision == 'defer':
        return {'status': 'deferred', 'reason': 'Needs human review'}
    return {'status': 'executed', 'context': context[:60], 'result': 'success'}


@pxt.udf
def flag_issues(reasoning: str, confidence: float) -> str:
    issues = []
    if confidence < 0.5:
        issues.append('Very low confidence — needs rethink')
    if confidence < 0.7:
        issues.append('Below threshold — verify reasoning')
    if 'High complexity' in reasoning:
        issues.append('Complex task — check edge cases')
    return '; '.join(issues) if issues else 'No issues flagged'
