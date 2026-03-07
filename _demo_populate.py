#!/usr/bin/env python3
"""Populate a fresh Pixeltable instance with rich demo data for the dashboard."""

import os, datetime as dt, random, math, hashlib

os.environ['PIXELTABLE_HOME'] = '/tmp/pxt_dashboard_demo'

import pixeltable as pxt

pxt.init()

random.seed(42)

# ── Cleanup ──────────────────────────────────────────────────────────────────
for d in ['demo']:
    try:
        pxt.drop_dir(d, force=True)
    except Exception:
        pass

# ── Directories ──────────────────────────────────────────────────────────────
dirs = [
    'demo',
    'demo.content',
    'demo.content.articles',
    'demo.content.docs',
    'demo.analytics',
    'demo.analytics.web',
    'demo.analytics.product',
    'demo.ml',
    'demo.ml.training',
    'demo.ml.evaluation',
    'demo.ecommerce',
    'demo.ecommerce.catalog',
    'demo.ecommerce.orders',
    'demo.ecommerce.reviews',
    'demo.social',
    'demo.social.posts',
    'demo.social.moderation',
]
for d in dirs:
    pxt.create_dir(d, if_exists='ignore')
print(f'✓ Created {len(dirs)} directories')


# ── UDFs ─────────────────────────────────────────────────────────────────────
@pxt.udf
def word_count(text: str) -> int:
    return len(text.split()) if text else 0


@pxt.udf
def char_count(text: str) -> int:
    return len(text) if text else 0


@pxt.udf
def slug(title: str) -> str:
    return '-'.join(title.lower().split()[:8]) if title else ''


@pxt.udf
def reading_time_min(text: str) -> float:
    return round(len(text.split()) / 200, 1) if text else 0.0


@pxt.udf
def sentiment_label(text: str) -> str:
    pos = ['great', 'excellent', 'love', 'best', 'amazing', 'perfect', 'fantastic']
    neg = ['bad', 'terrible', 'worst', 'awful', 'hate', 'poor', 'horrible']
    t = (text or '').lower()
    p = sum(1 for w in pos if w in t)
    n = sum(1 for w in neg if w in t)
    if p > n:
        return 'positive'
    if n > p:
        return 'negative'
    return 'neutral'


@pxt.udf
def extract_tags(text: str) -> str:
    keywords = [
        'AI',
        'ML',
        'data',
        'model',
        'training',
        'pipeline',
        'multimodal',
        'vector',
        'embedding',
        'RAG',
        'LLM',
        'image',
        'video',
        'audio',
    ]
    found = [k for k in keywords if k.lower() in (text or '').lower()]
    return ', '.join(found[:5]) if found else 'general'


@pxt.udf
def text_hash(text: str) -> str:
    return hashlib.md5((text or '').encode()).hexdigest()[:12]


@pxt.udf
def price_bucket(price: float) -> str:
    if price < 25:
        return 'budget'
    if price < 100:
        return 'mid-range'
    if price < 500:
        return 'premium'
    return 'luxury'


@pxt.udf
def engagement_score(likes: int, shares: int, comments: int) -> float:
    return round(likes * 1.0 + shares * 2.0 + comments * 1.5, 2)


@pxt.udf
def normalize_0_1(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.5
    return round((value - min_val) / (max_val - min_val), 4)


@pxt.udf
def log_transform(value: float) -> float:
    return round(math.log1p(abs(value)), 4)


print('✓ Registered 11 UDFs')


# ── 1. Articles ──────────────────────────────────────────────────────────────
articles = pxt.create_table(
    'demo.content.articles.posts',
    {'title': pxt.String, 'body': pxt.String, 'author': pxt.String, 'category': pxt.String, 'published': pxt.Timestamp},
)

article_data = [
    (
        'Getting Started with Pixeltable',
        'Pixeltable unifies data storage and AI orchestration in one declarative framework. With computed columns and automatic lineage tracking you can build multimodal AI pipelines without glue code. This tutorial walks you through creating your first table, adding computed columns, and querying results. The framework handles all the complexity of data versioning and dependency management behind the scenes.',
        'Alice',
        'tutorial',
    ),
    (
        'Building RAG Pipelines',
        'Retrieval-Augmented Generation combines document retrieval with language model generation. Pixeltable makes this easy with embedding indices and computed columns that chain together retrieval and generation steps. You can build a complete RAG pipeline in just a few lines of code, with automatic caching and incremental updates.',
        'Bob',
        'rag',
    ),
    (
        'Multimodal Data Management',
        'Managing images, videos, audio, and text together is challenging. Pixeltable provides a unified interface for all media types with automatic format handling and lazy loading. Store everything in one table and let computed columns handle the transformations.',
        'Charlie',
        'data',
    ),
    (
        'Vector Search at Scale',
        'Embedding indices in Pixeltable enable fast similarity search across millions of records. Combined with computed columns that generate embeddings automatically, you get a production-ready vector search system with zero infrastructure overhead.',
        'Alice',
        'ml',
    ),
    (
        'Computer Vision Pipelines',
        'Build end-to-end computer vision pipelines with Pixeltable. From image ingestion to model inference, everything is declarative. Add object detection, classification, or segmentation as computed columns and results are cached automatically.',
        'Diana',
        'vision',
    ),
    (
        'LLM Integration Patterns',
        'Integrating large language models into your data pipeline is straightforward with Pixeltable. Use computed columns to call OpenAI, Anthropic, or local models. Results are cached, versioned, and queryable just like any other column.',
        'Eve',
        'llm',
    ),
    (
        'Data Versioning Best Practices',
        'Every change in Pixeltable is versioned automatically. This guide covers how to leverage versioning for reproducible experiments, A/B testing, and rollback capabilities. Learn to use snapshots and views for different use cases.',
        'Bob',
        'data',
    ),
    (
        'Audio Processing with Pixeltable',
        'Process audio files at scale using Pixeltable computed columns. From transcription to speaker diarization, build complete audio analysis pipelines declaratively. Support for WAV, MP3, FLAC and more.',
        'Frank',
        'audio',
    ),
    (
        'Fine-tuning Models with Pixeltable',
        'Use Pixeltable as your data backend for model fine-tuning. Manage training data, track experiments, and version your datasets. Integration with PyTorch DataLoaders makes it easy to feed data into your training loop.',
        'Alice',
        'ml',
    ),
    (
        'Document Intelligence',
        'Extract structured data from PDFs, contracts, and reports using Pixeltable document processing pipelines. Combine OCR, layout analysis, and LLM extraction in a single declarative pipeline.',
        'Grace',
        'document',
    ),
    (
        'Real-time Data Pipelines',
        'Pixeltable supports incremental processing for real-time applications. As new data arrives, computed columns update automatically. Learn how to build streaming-like pipelines with batch efficiency.',
        'Charlie',
        'data',
    ),
    (
        'Sentiment Analysis at Scale',
        'Build a sentiment analysis pipeline that processes thousands of reviews per minute. Pixeltable handles batching, caching, and error recovery automatically so you can focus on the model logic.',
        'Diana',
        'nlp',
    ),
    (
        'Image Generation Workflows',
        'Manage image generation workflows with Pixeltable. Store prompts, parameters, and generated images together. Use computed columns to run DALL-E, Stable Diffusion, or Midjourney and track everything automatically.',
        'Eve',
        'vision',
    ),
    (
        'Multi-Agent Systems',
        'Orchestrate multi-agent AI systems with Pixeltable as the shared memory layer. Agents can read and write to tables, with computed columns triggering downstream processing automatically.',
        'Frank',
        'agents',
    ),
    (
        'Production Deployment Guide',
        'Deploy Pixeltable-powered applications to production. This guide covers FastAPI integration, scaling strategies, monitoring, and best practices for running Pixeltable in a production environment.',
        'Grace',
        'deployment',
    ),
]

articles.insert(
    [
        {
            'title': t,
            'body': b,
            'author': a,
            'category': c,
            'published': dt.datetime(2025, 1, 1) + dt.timedelta(days=i * 7),
        }
        for i, (t, b, a, c) in enumerate(article_data)
    ]
)

articles.add_computed_column(word_count=word_count(articles.body))
articles.add_computed_column(char_count=char_count(articles.body))
articles.add_computed_column(url_slug=slug(articles.title))
articles.add_computed_column(read_time=reading_time_min(articles.body))
articles.add_computed_column(tags=extract_tags(articles.body))
articles.add_computed_column(content_hash=text_hash(articles.body))
articles.add_computed_column(sentiment=sentiment_label(articles.body))
articles.add_computed_column(is_long_read=articles.word_count > 100)
print(f'✓ Articles: {articles.count()} rows, 8 computed columns')


# ── 2. Documentation Pages ──────────────────────────────────────────────────
pages = pxt.create_table(
    'demo.content.docs.pages', {'title': pxt.String, 'content': pxt.String, 'section': pxt.String, 'order_idx': pxt.Int}
)

sections = [
    'Getting Started',
    'Core Concepts',
    'API Reference',
    'Tutorials',
    'Deployment',
    'Advanced',
    'FAQ',
    'Changelog',
]
for si, section in enumerate(sections):
    pages.insert(
        [
            {
                'title': f'{section} - Part {j + 1}',
                'content': f'Documentation for {section.lower()} part {j + 1}. ' * 20,
                'section': section,
                'order_idx': si * 10 + j,
            }
            for j in range(4)
        ]
    )

pages.add_computed_column(words=word_count(pages.content))
pages.add_computed_column(page_hash=text_hash(pages.content))
print(f'✓ Doc pages: {pages.count()} rows')


# ── 3. Web Pageviews ────────────────────────────────────────────────────────
pageviews = pxt.create_table(
    'demo.analytics.web.pageviews',
    {
        'url': pxt.String,
        'referrer': pxt.String,
        'user_agent': pxt.String,
        'duration_sec': pxt.Float,
        'timestamp': pxt.Timestamp,
    },
)

urls = ['/home', '/docs', '/pricing', '/blog', '/api', '/dashboard', '/signup', '/login', '/about', '/contact']
referrers = ['google.com', 'twitter.com', 'github.com', 'direct', 'hackernews', 'reddit.com']
agents = ['Chrome/120', 'Firefox/121', 'Safari/17', 'Edge/120', 'curl/8.0']

pageviews.insert(
    [
        {
            'url': random.choice(urls),
            'referrer': random.choice(referrers),
            'user_agent': random.choice(agents),
            'duration_sec': round(random.expovariate(1 / 45), 1),
            'timestamp': dt.datetime(2025, 1, 1) + dt.timedelta(hours=random.randint(0, 4000)),
        }
        for _ in range(200)
    ]
)

pageviews.add_computed_column(is_bounce=pageviews.duration_sec < 10)
pageviews.add_computed_column(duration_log=log_transform(pageviews.duration_sec))
print(f'✓ Pageviews: {pageviews.count()} rows')


# ── 4. Daily Metrics ────────────────────────────────────────────────────────
metrics = pxt.create_table(
    'demo.analytics.web.daily_metrics',
    {
        'date': pxt.Date,
        'page': pxt.String,
        'visitors': pxt.Int,
        'pageviews': pxt.Int,
        'avg_duration': pxt.Float,
        'bounce_rate': pxt.Float,
    },
)

metrics.insert(
    [
        {
            'date': dt.date(2025, 1, 1) + dt.timedelta(days=i),
            'page': random.choice(urls),
            'visitors': random.randint(10, 500),
            'pageviews': random.randint(20, 1200),
            'avg_duration': round(random.uniform(5, 120), 1),
            'bounce_rate': round(random.uniform(0.1, 0.8), 3),
        }
        for i in range(90)
    ]
)

metrics.add_computed_column(engagement=metrics.pageviews * (1 - metrics.bounce_rate))
metrics.add_computed_column(visitors_log=log_transform(metrics.visitors))
print(f'✓ Daily metrics: {metrics.count()} rows')


# ── 5. Product Events ───────────────────────────────────────────────────────
events = pxt.create_table(
    'demo.analytics.product.events',
    {'event_type': pxt.String, 'user_id': pxt.String, 'properties': pxt.Json, 'timestamp': pxt.Timestamp},
)

event_types = ['page_view', 'button_click', 'form_submit', 'signup', 'login', 'purchase', 'error', 'search']

events.insert(
    [
        {
            'event_type': random.choice(event_types),
            'user_id': f'user_{random.randint(1, 50):03d}',
            'properties': {'screen': random.choice(['home', 'settings', 'profile']), 'value': random.randint(1, 100)},
            'timestamp': dt.datetime(2025, 3, 1) + dt.timedelta(minutes=random.randint(0, 50000)),
        }
        for _ in range(300)
    ]
)
print(f'✓ Product events: {events.count()} rows')


# ── 6. ML Training Runs ─────────────────────────────────────────────────────
runs = pxt.create_table(
    'demo.ml.training.runs',
    {
        'model_name': pxt.String,
        'run_id': pxt.String,
        'epoch': pxt.Int,
        'loss': pxt.Float,
        'accuracy': pxt.Float,
        'learning_rate': pxt.Float,
        'batch_size': pxt.Int,
        'started_at': pxt.Timestamp,
    },
)

models = ['resnet50', 'bert-base', 'clip-vit', 'whisper-small', 'yolov8']
for m in models:
    base_acc = random.uniform(0.6, 0.85)
    for epoch in range(16):
        runs.insert(
            [
                {
                    'model_name': m,
                    'run_id': f'{m}-run-{random.randint(100, 999)}',
                    'epoch': epoch,
                    'loss': round(max(0.01, 2.0 - epoch * 0.12 + random.gauss(0, 0.1)), 4),
                    'accuracy': round(min(0.99, base_acc + epoch * 0.02 + random.gauss(0, 0.01)), 4),
                    'learning_rate': 0.001 * (0.95**epoch),
                    'batch_size': random.choice([16, 32, 64]),
                    'started_at': dt.datetime(2025, 2, 1) + dt.timedelta(hours=epoch * 2),
                }
            ]
        )

runs.add_computed_column(loss_log=log_transform(runs.loss))
runs.add_computed_column(is_converged=runs.loss < 0.5)
runs.add_computed_column(lr_log=log_transform(runs.learning_rate))
print(f'✓ Training runs: {runs.count()} rows')


# ── 7. Evaluation Results ────────────────────────────────────────────────────
results = pxt.create_table(
    'demo.ml.evaluation.results',
    {
        'model_name': pxt.String,
        'dataset': pxt.String,
        'metric': pxt.String,
        'value': pxt.Float,
        'evaluated_at': pxt.Timestamp,
    },
)

datasets = ['train', 'val', 'test', 'holdout', 'adversarial']
eval_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

results.insert(
    [
        {
            'model_name': random.choice(models),
            'dataset': random.choice(datasets),
            'metric': random.choice(eval_metrics),
            'value': round(random.uniform(0.5, 0.99), 4),
            'evaluated_at': dt.datetime(2025, 3, 1) + dt.timedelta(hours=random.randint(0, 720)),
        }
        for _ in range(250)
    ]
)

results.add_computed_column(is_good=results.value > 0.8)
print(f'✓ Evaluation results: {results.count()} rows')


# ── 8. E-commerce Products ──────────────────────────────────────────────────
products = pxt.create_table(
    'demo.ecommerce.catalog.products',
    {
        'name': pxt.String,
        'description': pxt.String,
        'category': pxt.String,
        'price': pxt.Float,
        'stock': pxt.Int,
        'rating': pxt.Float,
    },
)

categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Toys', 'Food']
adjectives = ['Smart', 'Premium', 'Classic', 'Ultra', 'Eco', 'Compact', 'Pro', 'Deluxe']
nouns = ['Widget', 'Gadget', 'Tool', 'Device', 'Kit', 'Set', 'Pack', 'Bundle']

products.insert(
    [
        {
            'name': f'{random.choice(adjectives)} {random.choice(nouns)} {i + 1}',
            'description': f'A great {random.choice(categories).lower()} product with excellent quality and fast delivery. '
            f'Best value for professionals and enthusiasts alike. Made with premium materials.',
            'category': random.choice(categories),
            'price': round(random.uniform(5, 999), 2),
            'stock': random.randint(0, 500),
            'rating': round(random.uniform(1, 5), 1),
        }
        for i in range(120)
    ]
)

products.add_computed_column(bucket=price_bucket(products.price))
products.add_computed_column(desc_words=word_count(products.description))
products.add_computed_column(desc_sentiment=sentiment_label(products.description))
products.add_computed_column(desc_hash=text_hash(products.description))
print(f'✓ Products: {products.count()} rows')


# ── 9. Orders ────────────────────────────────────────────────────────────────
orders = pxt.create_table(
    'demo.ecommerce.orders.transactions',
    {
        'order_id': pxt.String,
        'customer_id': pxt.String,
        'product_name': pxt.String,
        'quantity': pxt.Int,
        'unit_price': pxt.Float,
        'status': pxt.String,
        'ordered_at': pxt.Timestamp,
    },
)

statuses = ['pending', 'shipped', 'delivered', 'returned', 'cancelled']

orders.insert(
    [
        {
            'order_id': f'ORD-{10000 + i}',
            'customer_id': f'CUST-{random.randint(1, 80):04d}',
            'product_name': f'{random.choice(adjectives)} {random.choice(nouns)} {random.randint(1, 120)}',
            'quantity': random.randint(1, 5),
            'unit_price': round(random.uniform(10, 500), 2),
            'status': random.choice(statuses),
            'ordered_at': dt.datetime(2025, 1, 1) + dt.timedelta(hours=random.randint(0, 3000)),
        }
        for i in range(200)
    ]
)

orders.add_computed_column(total=orders.quantity * orders.unit_price)
orders.add_computed_column(total_log=log_transform(orders.quantity * orders.unit_price))
print(f'✓ Orders: {orders.count()} rows')


# ── 10. Reviews ──────────────────────────────────────────────────────────────
reviews = pxt.create_table(
    'demo.ecommerce.reviews.feedback',
    {
        'product_name': pxt.String,
        'reviewer': pxt.String,
        'rating': pxt.Int,
        'text': pxt.String,
        'helpful_votes': pxt.Int,
        'posted_at': pxt.Timestamp,
    },
)

review_templates = [
    'Absolutely love this product! Best purchase I made this year. Excellent quality.',
    'Decent product for the price. Nothing special but gets the job done.',
    'Terrible quality. Broke after one week. Would not recommend to anyone.',
    'Great value for money. Fast shipping and excellent packaging.',
    'Average product. Expected better quality based on the description.',
    'Amazing! Exceeded all my expectations. Perfect for daily use.',
    'Not worth the price. Poor build quality and bad customer service.',
    'Fantastic product! Will definitely buy again. Highly recommended.',
    'Horrible experience. The product arrived damaged and support was awful.',
    'Good product overall. Minor issues but nothing deal-breaking.',
]

reviews.insert(
    [
        {
            'product_name': f'{random.choice(adjectives)} {random.choice(nouns)} {random.randint(1, 120)}',
            'reviewer': f'User{random.randint(1, 200)}',
            'rating': random.randint(1, 5),
            'text': random.choice(review_templates),
            'helpful_votes': random.randint(0, 50),
            'posted_at': dt.datetime(2025, 2, 1) + dt.timedelta(hours=random.randint(0, 2000)),
        }
        for _ in range(100)
    ]
)

reviews.add_computed_column(sentiment=sentiment_label(reviews.text))
reviews.add_computed_column(review_words=word_count(reviews.text))
reviews.add_computed_column(review_hash=text_hash(reviews.text))
print(f'✓ Reviews: {reviews.count()} rows')


# ── 11. Social Posts ─────────────────────────────────────────────────────────
community = pxt.create_table(
    'demo.social.posts.community',
    {
        'author': pxt.String,
        'content': pxt.String,
        'likes': pxt.Int,
        'shares': pxt.Int,
        'comments': pxt.Int,
        'platform': pxt.String,
        'posted_at': pxt.Timestamp,
    },
)

platforms = ['twitter', 'linkedin', 'reddit', 'discord', 'mastodon']
post_templates = [
    'Just shipped a new feature using Pixeltable. The computed columns are amazing!',
    'Exploring multimodal AI pipelines with Pixeltable. So much easier than expected.',
    'Built a complete RAG system in 20 lines of code. Pixeltable handles all the complexity.',
    'Anyone else using Pixeltable for their ML workflows? Game changer for data management.',
    'New blog post: How we reduced our pipeline complexity by 80% with Pixeltable.',
    'Pixeltable + FastAPI = the perfect stack for multimodal AI applications.',
    'Just discovered Pixeltable embedding indices. Vector search has never been this easy.',
    'Our team switched from custom ETL to Pixeltable. Best decision we made this quarter.',
]

community.insert(
    [
        {
            'author': f'@user{random.randint(1, 100)}',
            'content': random.choice(post_templates),
            'likes': random.randint(0, 500),
            'shares': random.randint(0, 100),
            'comments': random.randint(0, 80),
            'platform': random.choice(platforms),
            'posted_at': dt.datetime(2025, 3, 1) + dt.timedelta(hours=random.randint(0, 1500)),
        }
        for _ in range(150)
    ]
)

community.add_computed_column(engagement=engagement_score(community.likes, community.shares, community.comments))
community.add_computed_column(post_sentiment=sentiment_label(community.content))
community.add_computed_column(post_words=word_count(community.content))
community.add_computed_column(post_tags=extract_tags(community.content))
community.add_computed_column(is_viral=community.likes > 200)
print(f'✓ Social posts: {community.count()} rows')


# ── 12. Moderation Flags ────────────────────────────────────────────────────
flags = pxt.create_table(
    'demo.social.moderation.flags',
    {
        'post_id': pxt.Int,
        'reason': pxt.String,
        'severity': pxt.String,
        'reviewed': pxt.Bool,
        'flagged_at': pxt.Timestamp,
    },
)

reasons = ['spam', 'harassment', 'misinformation', 'off-topic', 'self-promotion', 'hate-speech']
severities = ['low', 'medium', 'high', 'critical']

flags.insert(
    [
        {
            'post_id': random.randint(1, 150),
            'reason': random.choice(reasons),
            'severity': random.choice(severities),
            'reviewed': random.random() > 0.3,
            'flagged_at': dt.datetime(2025, 3, 1) + dt.timedelta(hours=random.randint(0, 1000)),
        }
        for _ in range(60)
    ]
)
print(f'✓ Moderation flags: {flags.count()} rows')


# ── Views ────────────────────────────────────────────────────────────────────
def safe_view(name, query):
    try:
        pxt.create_view(name, query)
        return True
    except Exception as e:
        print(f'  ⚠ View {name}: {str(e)[:80]}')
        return False


created_views = 0
created_views += safe_view('demo.content.articles.long_reads', articles.where(articles.word_count > 80))
created_views += safe_view('demo.content.articles.tutorials', articles.where(articles.category == 'tutorial'))
created_views += safe_view('demo.analytics.web.bounces', pageviews.where(pageviews.duration_sec < 10))
created_views += safe_view('demo.analytics.web.engaged', pageviews.where(pageviews.duration_sec > 60))
created_views += safe_view('demo.ml.training.converged', runs.where(runs.loss < 0.5))
created_views += safe_view('demo.ml.evaluation.good_models', results.where(results.value > 0.85))
created_views += safe_view('demo.ecommerce.catalog.premium', products.where(products.price > 200))
created_views += safe_view('demo.ecommerce.catalog.budget', products.where(products.price < 30))
created_views += safe_view('demo.ecommerce.catalog.out_of_stock', products.where(products.stock == 0))
created_views += safe_view('demo.ecommerce.reviews.positive', reviews.where(reviews.rating >= 4))
created_views += safe_view('demo.ecommerce.reviews.negative', reviews.where(reviews.rating <= 2))
created_views += safe_view('demo.social.posts.viral', community.where(community.likes > 200))
created_views += safe_view('demo.social.posts.linkedin', community.where(community.platform == 'linkedin'))
created_views += safe_view('demo.social.moderation.unreviewed', flags.where(flags.reviewed == False))
created_views += safe_view('demo.social.moderation.critical', flags.where(flags.severity == 'critical'))
print(f'✓ Created {created_views} views')


# ── Summary ──────────────────────────────────────────────────────────────────
all_dirs = pxt.list_dirs('', recursive=True)
all_tables = pxt.list_tables('', recursive=True)

total_rows = 0
total_computed = 0
total_views = 0
for t in all_tables:
    try:
        tbl = pxt.get_table(t)
        md = tbl.get_metadata()
        total_rows += tbl.count()
        total_computed += sum(1 for c in md['columns'].values() if c.get('computed_with'))
        if md.get('is_view'):
            total_views += 1
    except Exception:
        pass

print(f"""
{'=' * 60}
  Dashboard Demo Summary
{'=' * 60}
  Directories:      {len(all_dirs)}
  Base tables:      {len(all_tables) - total_views}
  Views:            {total_views}
  Total rows:       {total_rows:,}
  Computed columns: {total_computed}
  UDFs:             11
{'=' * 60}
  Dashboard: http://localhost:8080
{'=' * 60}
""")
