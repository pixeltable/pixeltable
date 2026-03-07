"""
Complex multi-pipeline demo — vision, NLP, and agent workflows.
Creates deep dependency chains, multiple views, and cross-column lineage
to stress-test the dashboard lineage graph.
"""

import pixeltable as pxt
import datetime as dt
import shutil
import os

from _pipeline_udfs import (
    # Vision
    detect_objects,
    count_detections,
    check_person,
    classify_scene,
    assess_risk,
    estimate_pose,
    classify_action,
    # NLP
    detect_language,
    analyze_sentiment,
    extract_entities,
    summarize_text,
    classify_topic,
    score_quality,
    compute_readability,
    extract_phrases,
    # Agent
    parse_intent,
    assess_complexity,
    generate_plan,
    chain_of_thought,
    evaluate_confidence,
    make_decision,
    execute_action,
    flag_issues,
)

DB_DIR = '/tmp/pxt_pipeline_demo'

# ── Fresh start ──────────────────────────────────────────────────────────────

if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)
    print(f'Cleaned {DB_DIR}')

os.environ['PIXELTABLE_HOME'] = DB_DIR
pxt.init()
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. VISION PIPELINE
#    frames → objects_json → num_objects → risk_score
#                          → has_person → [person_frames view] → pose → action
#                          → scene_type ↗
# ═══════════════════════════════════════════════════════════════════════════════

print('━━━ Vision Pipeline ━━━')
pxt.create_dir('vision')

frames = pxt.create_table(
    'vision/frames',
    {'frame_id': pxt.Int, 'image_url': pxt.String, 'source_video': pxt.String, 'timestamp_sec': pxt.Float},
)

# Chain: image_url → detect_objects → objects_json
frames.add_computed_column(objects_json=detect_objects(frames.image_url))
# Fan-out from objects_json
frames.add_computed_column(num_objects=count_detections(frames.objects_json))
frames.add_computed_column(has_person=check_person(frames.objects_json))
# Fan-in: image_url + objects_json → scene_type
frames.add_computed_column(scene_type=classify_scene(frames.image_url, frames.objects_json))
# Deep chain: scene_type + num_objects → risk_score
frames.add_computed_column(risk_score=assess_risk(frames.scene_type, frames.num_objects))

image_urls = [f'https://picsum.photos/id/{i}/640/480' for i in range(10, 22)]
frames.insert(
    [
        {
            'frame_id': i,
            'image_url': image_urls[i],
            'source_video': f'cam_{(i % 3) + 1}.mp4',
            'timestamp_sec': round(i * 0.5, 1),
        }
        for i in range(12)
    ]
)
print(f'  frames: {frames.count()} rows, 5 computed cols')

# View: person_frames (where person detected) + extra computed cols
person_frames = pxt.create_view(
    'vision/person_frames',
    frames.where(frames.has_person == True),  # noqa: E712
)
person_frames.add_computed_column(pose_label=estimate_pose(person_frames.image_url))
person_frames.add_computed_column(action=classify_action(person_frames.pose_label, person_frames.scene_type))
print(f'  person_frames (view): {person_frames.count()} rows, +2 computed cols')

# View: high-risk alerts
alerts = pxt.create_view('vision/alerts', frames.where(frames.risk_score > 0.6))
print(f'  alerts (view): {alerts.count()} rows')

# ═══════════════════════════════════════════════════════════════════════════════
# 2. NLP PIPELINE
#    body → language, sentiment, entities, summary
#    title + body → topic
#    body + sentiment + entities → quality_score
#    [english view] body → readability; summary + entities → key_phrases
#    [flagged view] quality_score < 0.5
# ═══════════════════════════════════════════════════════════════════════════════

print('\n━━━ NLP Pipeline ━━━')
pxt.create_dir('nlp')

corpus = pxt.create_table(
    'nlp/corpus',
    {'doc_id': pxt.Int, 'title': pxt.String, 'body': pxt.String, 'source': pxt.String, 'created_at': pxt.Timestamp},
)

# Fan-out from body
corpus.add_computed_column(language=detect_language(corpus.body))
corpus.add_computed_column(sentiment=analyze_sentiment(corpus.body))
corpus.add_computed_column(entities_json=extract_entities(corpus.body))
corpus.add_computed_column(summary=summarize_text(corpus.body))
# Fan-in: title + body → topic
corpus.add_computed_column(topic=classify_topic(corpus.title, corpus.body))
# Deep fan-in: body + sentiment + entities → quality_score
corpus.add_computed_column(quality_score=score_quality(corpus.body, corpus.sentiment, corpus.entities_json))

articles = [
    (
        'Advances in Neural Architecture Search',
        'Recent research in neural architecture search has produced amazing results. '
        'AutoML systems can now design networks that outperform hand-crafted architectures. '
        'Google Brain and DeepMind have published several excellent studies on this topic. '
        'The implications for AI research are profound and wonderful.',
        'ArXiv',
    ),
    (
        'The Future of Multimodal AI',
        'Pixeltable is building the data layer for multimodal AI pipelines. '
        'By unifying data storage and AI orchestration, it eliminates glue code. '
        'The declarative approach to computed columns and automatic lineage tracking '
        'makes complex machine learning workflows much simpler to manage.',
        'Blog',
    ),
    (
        'Clinical Trial Results for New Treatment',
        'A randomized clinical study of 500 patients showed significant improvement. '
        'The treatment group had 40% better outcomes than the control group. '
        'Dr. Smith from Johns Hopkins led the research team. '
        'These findings could revolutionize patient care.',
        'PubMed',
    ),
    (
        'Quarterly Earnings Report: TechCorp',
        'TechCorp reported record revenue of $5.2B this quarter. '
        'The company growth was driven by strong demand for cloud services. '
        'CEO Martinez announced plans to invest $1B in AI research. '
        'Market analysts expect continued growth through next year.',
        'Reuters',
    ),
    (
        'Physics of Black Holes',
        'New observations from the James Webb Space Telescope reveal surprising structure. '
        'The discovery challenges existing models of black hole formation. '
        'Professor Chen published the hypothesis in Nature this week.',
        'Nature',
    ),
    (
        'Why This Startup Failed',
        'The company had terrible leadership and poor execution. '
        'Their worst decision was ignoring customer feedback. '
        'The product was boring and failed to attract users. '
        'Investors lost confidence after the bad quarterly report.',
        'TechCrunch',
    ),
    (
        '简介机器学习在医疗领域的应用',
        '机器学习技术正在改变医疗诊断的方式。深度学习模型可以从医学影像中识别疾病。',
        'CNKI',
    ),
    (
        'Algorithmic Trading Strategies',
        'Modern algorithmic trading relies on machine learning models. '
        'Data pipelines process millions of market events per second. '
        'The best strategies combine multiple signals and risk management. '
        'Software systems must handle extreme latency requirements.',
        'Quantitative Finance',
    ),
    (
        'Edge Computing for IoT',
        'Edge computing brings AI model inference closer to the data source. '
        'This reduces latency and bandwidth for IoT sensor networks. '
        'TinyML frameworks enable neural network deployment on microcontrollers. '
        'The technology is excellent for real-time object detection applications.',
        'IEEE',
    ),
    ('Bad.', 'Bad.', 'Unknown'),
]

corpus.insert(
    [
        {
            'doc_id': i,
            'title': title,
            'body': body,
            'source': source,
            'created_at': dt.datetime(2025, 1, 1) + dt.timedelta(days=i * 12),
        }
        for i, (title, body, source) in enumerate(articles)
    ]
)
print(f'  corpus: {corpus.count()} rows, 6 computed cols')

# View: English documents + extra NLP
english = pxt.create_view('nlp/english', corpus.where(corpus.language == 'en'))
english.add_computed_column(readability=compute_readability(english.body))
english.add_computed_column(key_phrases=extract_phrases(english.summary, english.entities_json))
print(f'  english (view): {english.count()} rows, +2 computed cols')

# View: Flagged low-quality
flagged = pxt.create_view('nlp/flagged', corpus.where(corpus.quality_score < 0.5))
print(f'  flagged (view): {flagged.count()} rows')

# ═══════════════════════════════════════════════════════════════════════════════
# 3. AGENT PIPELINE  (deepest chain: 7 levels)
#    instruction → parsed_intent ──→ plan ──→ reasoning ──→ confidence ──→ decision ──→ execution_log
#    context ──────────────────────↗         ↗                              ↗
#                    complexity ────────────╯              context ────────╯
# ═══════════════════════════════════════════════════════════════════════════════

print('\n━━━ Agent Pipeline ━━━')
pxt.create_dir('agent')

tasks = pxt.create_table(
    'agent/tasks',
    {
        'task_id': pxt.String,
        'instruction': pxt.String,
        'context': pxt.String,
        'priority': pxt.Int,
        'created_at': pxt.Timestamp,
    },
)

# Level 1: instruction → parsed_intent
tasks.add_computed_column(parsed_intent=parse_intent(tasks.instruction))
# Level 1b: instruction + context → complexity
tasks.add_computed_column(complexity=assess_complexity(tasks.instruction, tasks.context))
# Level 2: parsed_intent + context → plan
tasks.add_computed_column(plan=generate_plan(tasks.parsed_intent, tasks.context))
# Level 3: plan + complexity → reasoning (deep chain)
tasks.add_computed_column(reasoning=chain_of_thought(tasks.plan, tasks.complexity))
# Level 4: reasoning → confidence (may throw errors)
tasks.add_computed_column(confidence=evaluate_confidence(tasks.reasoning))
# Level 5: reasoning + confidence → decision
tasks.add_computed_column(decision=make_decision(tasks.reasoning, tasks.confidence))
# Level 6: decision + context → execution_log
tasks.add_computed_column(execution_log=execute_action(tasks.decision, tasks.context))

task_data = [
    (
        'T-001',
        'Search for all documents related to machine learning in healthcare',
        'We have a corpus of 10K medical papers and clinical notes. Need to find relevant ML applications.',
        3,
    ),
    (
        'T-002',
        'Create a summary report of quarterly earnings for all portfolio companies',
        'Portfolio includes TechCorp, BioGen, and DataVault. Q3 2025 financials are available.',
        2,
    ),
    (
        'T-003',
        'Analyze sentiment trends across customer feedback channels',
        'Customer reviews from App Store, Google Play, and Zendesk tickets. Last 90 days.',
        1,
    ),
    (
        'T-004',
        'Update the product recommendation model with new user data and then retrain',
        'New batch of 50K user interactions collected. Current model accuracy is 78%. Need to incorporate '
        'multiple data sources and validate against holdout set after retraining.',
        1,
    ),
    (
        'T-005',
        'Find and fix all broken image URLs in the content database',
        'CMS database with 25K articles. Some images point to deprecated CDN.',
        2,
    ),
    (
        'T-006',
        'Generate a comprehensive competitive analysis report',
        'Competitors: AlphaAI, BetaData, GammaML. Need pricing, features, market share analysis.',
        1,
    ),
    (
        'T-007',
        'Delete all expired user sessions and clean up temporary files',
        'Redis cache has 100K session keys. File storage has /tmp artifacts older than 30 days.',
        3,
    ),
    (
        'T-008',
        'Evaluate the performance of all deployed models across every production endpoint',
        'We have 12 models serving predictions. Need latency, accuracy, and drift metrics for '
        'multiple time windows. Also assess resource utilization and cost efficiency after examining '
        'all the monitoring dashboards and alerting thresholds currently in place.',
        1,
    ),
    ('T-009', 'x', 'y', 3),
    (
        'T-010',
        'Analyze and then update the data pipeline to handle new sensor types and also '
        'generate documentation for the changes after reviewing every existing connector',
        'IoT platform with 50 sensor types. Adding 10 new industrial sensors. Each requires '
        'custom parsing, validation, and transformation logic. Must maintain backward compatibility.',
        1,
    ),
]

tasks.insert(
    [
        {
            'task_id': tid,
            'instruction': instr,
            'context': ctx,
            'priority': pri,
            'created_at': dt.datetime(2025, 6, 1) + dt.timedelta(hours=i * 6),
        }
        for i, (tid, instr, ctx, pri) in enumerate(task_data)
    ],
    on_error='ignore',
)
print(f'  tasks: {tasks.count()} rows, 7 computed cols (deepest chain: 6 levels)')

# View: completed (decision != reject)
completed = pxt.create_view('agent/completed', tasks.where(tasks.decision != 'reject'))
print(f'  completed (view): {completed.count()} rows')

# View: needs_review (confidence < 0.7) + extra computed col
needs_review = pxt.create_view('agent/needs_review', tasks.where(tasks.confidence < 0.7))
needs_review.add_computed_column(reviewer_notes=flag_issues(needs_review.reasoning, needs_review.confidence))
print(f'  needs_review (view): {needs_review.count()} rows, +1 computed col')

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '═' * 60)
print('Pipeline demo ready!')
print(f'  3 directories, 3 base tables, 5 views')
print(f'  ~20 computed columns with deep dependency chains')
print(f'  Dashboard: http://localhost:8080')
print('═' * 60)

import time

print('\nKeeping server alive (Ctrl+C to stop)...')
time.sleep(999999)
