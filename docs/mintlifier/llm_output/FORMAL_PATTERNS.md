# Formalized Pixeltable Patterns

## Pattern 1: Progressive Refinement Pipeline

### Where It Appears (but isn't named)
- Video detection: video → frames → detections → filtered_detections → alerts
- RAG: documents → chunks → embeddings → filtered_chunks → responses
- Audio: video → audio → transcription → sentences → searchable_segments

### The Pattern Structure

```python
# PATTERN: Progressive Refinement Pipeline
# Purpose: Transform raw data through increasingly refined stages,
#          with each stage queryable and reusable

def progressive_refinement_pipeline(source_table, stages):
    """
    Each stage:
    1. Can be queried independently
    2. Updates automatically when source changes
    3. Can branch to multiple refinements
    4. Maintains full lineage
    """
    current = source_table
    artifacts = {'source': source_table}
    
    for stage_name, transformation in stages:
        if 'iterator' in transformation:
            # One-to-many expansion (frames, chunks, sentences)
            current = pxt.create_view(
                f'{source_table.name}.{stage_name}',
                current,
                iterator=transformation['iterator']
            )
        else:
            # One-to-one refinement (computed column)
            current.add_computed_column(
                **{stage_name: transformation['function']}
            )
        
        artifacts[stage_name] = current
    
    return artifacts

# EXAMPLE INSTANTIATION:
video_pipeline = progressive_refinement_pipeline(
    videos_table,
    stages=[
        ('frames', {'iterator': FrameIterator(video=videos.video, fps=1)}),
        ('detections', {'function': yolox(frames.frame)}),
        ('people_only', {'function': filter_detections(frames.detections, 'person')}),
        ('alerts', {'function': create_alert_if(frames.people_only, threshold=5)})
    ]
)
```

### Why This Pattern Matters
1. **Not Obvious from Single Notebook**: Each notebook shows a linear pipeline, but the PATTERN is that you can branch, query, and reuse any stage
2. **Compositional Power**: Stages can be mixed and matched across use cases
3. **Debugging Paradise**: Query any intermediate stage to see what's happening
4. **Reusability**: Other pipelines can reference any stage

---

## Pattern 2: Multi-Model Consensus

### Where It Appears (fragmented)
- Object detection: yolox_tiny vs yolox_m vs yolox_x
- LLM comparison: OpenAI vs Anthropic vs Mistral
- Embeddings: MiniLM vs CLIP vs OpenAI

### The Pattern Structure

```python
# PATTERN: Multi-Model Consensus
# Purpose: Run multiple models on same data for comparison,
#          consensus, or ensemble predictions

def multi_model_consensus(table, input_column, models, consensus_fn=None):
    """
    Run N models on same input, optionally combine results
    """
    # Phase 1: Run all models in parallel
    for model_name, model_fn in models.items():
        table.add_computed_column(
            **{f'{model_name}_output': model_fn(input_column)}
        )
    
    # Phase 2: Extract comparable metrics
    for model_name in models.keys():
        table.add_computed_column(
            **{f'{model_name}_confidence': extract_confidence(table[f'{model_name}_output'])}
        )
    
    # Phase 3: Optional consensus
    if consensus_fn:
        model_outputs = [table[f'{m}_output'] for m in models.keys()]
        table.add_computed_column(
            consensus=consensus_fn(*model_outputs)
        )
    
    # Phase 4: Analysis columns
    table.add_computed_column(
        agreement_score=calculate_agreement([table[f'{m}_output'] for m in models.keys()]),
        best_model=select_best([table[f'{m}_confidence'] for m in models.keys()])
    )

# EXAMPLE: Never shown explicitly but implied across notebooks
multi_model_consensus(
    frames,
    frames.frame,
    models={
        'yolox_tiny': lambda x: yolox(x, model_id='yolox_tiny'),
        'yolox_m': lambda x: yolox(x, model_id='yolox_m'),
        'yolox_x': lambda x: yolox(x, model_id='yolox_x')
    },
    consensus_fn=majority_vote
)
```

### Hidden Insight
No single notebook shows this, but the pattern enables:
- A/B testing in production
- Gradual model migration
- Ensemble predictions
- Cost/quality optimization

---

## Pattern 3: Semantic Index Cascade

### Where It Appears (never explicitly connected)
- Embeddings notebook: create index
- RAG notebook: different chunking + embeddings
- Audio notebook: transcription + embedding

### The Pattern Structure

```python
# PATTERN: Semantic Index Cascade
# Purpose: Build multiple semantic indexes at different granularities
#          for the same content

def semantic_index_cascade(source_table, content_column, granularities):
    """
    Create multiple searchable views at different detail levels
    """
    indexes = {}
    
    for level_name, config in granularities.items():
        # Create view at this granularity
        view = pxt.create_view(
            f'{source_table.name}_{level_name}',
            source_table,
            iterator=config['splitter']
        )
        
        # Add embedding at this level
        view.add_computed_column(
            embedding=config['embedding_fn'](view.text)
        )
        
        # Create semantic index
        view.add_embedding_index(
            'text',
            embedding=config['embedding_fn']
        )
        
        indexes[level_name] = view
    
    # Create cross-level search function
    def search_all_levels(query, top_k=5):
        results = {}
        for level_name, view in indexes.items():
            results[level_name] = view.order_by(
                view.text.similarity(query),
                asc=False
            ).limit(top_k).select(view.text, view.source_doc)
        return results
    
    return indexes, search_all_levels

# EXAMPLE: Implied across RAG + embedding notebooks
indexes, search = semantic_index_cascade(
    documents,
    documents.content,
    granularities={
        'sentence': {
            'splitter': DocumentSplitter(separators='sentence'),
            'embedding_fn': sentence_transformer
        },
        'paragraph': {
            'splitter': DocumentSplitter(separators='paragraph'),
            'embedding_fn': sentence_transformer
        },
        'section': {
            'splitter': DocumentSplitter(separators='heading'),
            'embedding_fn': openai.embeddings
        }
    }
)
```

### Why This Is Powerful
- **Never Shown Together**: Notebooks show single-level indexing
- **Enables**: Multi-resolution search (overview → detail)
- **Hidden Benefit**: Can search at conversation level, then drill to sentence

---

## Pattern 4: Incremental Evaluation Loop

### Appears In (but never named)
- Object detection: evaluation metrics
- RAG: ground truth comparison
- Never connected as a pattern!

### The Pattern Structure

```python
# PATTERN: Incremental Evaluation Loop
# Purpose: Continuously evaluate model performance as new data arrives

def incremental_evaluation_loop(
    prediction_table,
    prediction_col,
    ground_truth_source,
    metrics
):
    """
    As new predictions are made, automatically compute metrics
    """
    # Step 1: Add ground truth column (might be delayed)
    prediction_table.add_computed_column(
        ground_truth=ground_truth_source(prediction_table.input)
    )
    
    # Step 2: Add evaluation columns
    for metric_name, metric_fn in metrics.items():
        prediction_table.add_computed_column(
            **{metric_name: metric_fn(
                prediction_table[prediction_col],
                prediction_table.ground_truth
            )}
        )
    
    # Step 3: Add aggregate metrics view
    metrics_view = pxt.create_view(
        f'{prediction_table.name}_metrics',
        prediction_table,
        is_snapshot=True  # Snapshot for performance
    )
    
    # Step 4: Add drift detection
    metrics_view.add_computed_column(
        drift_score=calculate_drift(
            metrics_view.select(window='1d'),
            metrics_view.select(window='7d')
        )
    )
    
    # Step 5: Alerting
    metrics_view.add_computed_column(
        alert=pxt.when(metrics_view.drift_score > 0.1)
              .then('Model drift detected!')
              .otherwise(None)
    )

# EXAMPLE: Implied but never explicit
incremental_evaluation_loop(
    predictions,
    'model_output',
    ground_truth_source=human_labels_when_available,
    metrics={
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }
)
```

---

## Pattern 5: The Versioned Experiment Table

### Never Explicitly Shown But Implied Everywhere

```python
# PATTERN: Versioned Experiment Table
# Purpose: Every table is actually an experiment that can be versioned

def versioned_experiment(name, setup_fn, versions_to_keep=3):
    """
    Pattern that emerges from how everyone uses Pixeltable
    """
    # Everyone does this:
    pxt.drop_table(name, ignore_errors=True)  # <-- IMPLIED PATTERN!
    
    # What they're really doing:
    version = get_next_version(name)
    table_name = f'{name}_v{version}'
    
    # Create versioned table
    table = setup_fn(table_name)
    
    # Link as "latest"
    create_alias(name, table_name)
    
    # Cleanup old versions
    cleanup_old_versions(name, versions_to_keep)
    
    return table

# This pattern is why everyone starts with:
# pxt.drop_dir('demo', force=True)  <-- They're versioning without knowing it!
```

---

## The Meta-Pattern: Declarative Incremental Permanence (DIP)

This is the philosophical pattern underlying everything:

```python
# META-PATTERN: Declarative Incremental Permanence
# You DECLARE what should exist
# It INCREMENTALLY updates 
# It's PERMANENT until explicitly changed

# Traditional imperative:
for video in videos:
    frames = extract_frames(video)
    detections = run_detection(frames)
    save_results(detections)

# Pixeltable DIP:
frames_view = pxt.create_view(iterator=FrameIterator(videos.video))
frames_view.add_computed_column(detections=yolox(frames.frame))
# Done. Forever. Automatically updates. Queryable. Permanent.
```

---

## Patterns That Only Emerge from Multiple Notebooks

### The "Goldilocks Chunking" Pattern
Seen across RAG, audio, and video notebooks but never named:

```python
# Too small: Lost context (word-level)
# Too large: Lost precision (document-level)  
# Just right: Multiple granularities simultaneously

# The pattern everyone converges to:
for chunk_size in [128, 512, 2048]:
    create_view_with_chunking(chunk_size)
    add_embeddings()
    test_retrieval_quality()
    
# Keep the best 2-3 granularities in production
```

### The "Cascade Cache" Pattern
Never mentioned but appears everywhere:

```python
# Expensive operations create cascading caches
table.add_computed_column(expensive=api_call())
table.add_computed_column(derived1=process(table.expensive))
table.add_computed_column(derived2=analyze(table.expensive))
# 'expensive' is computed once, cached, reused
# This pattern naturally emerges from the design!
```

---

## Why These Patterns Matter

1. **They're Not Documented** - Users discover them through experimentation
2. **They're Compositional** - Patterns combine into bigger patterns
3. **They're Emergent** - The design of Pixeltable makes them natural
4. **They're Powerful** - Each pattern unlocks multiple use cases

## Recommendation: Pattern Library

Create a formal pattern library with:
```python
from pixeltable.patterns import (
    ProgressiveRefinement,
    MultiModelConsensus,
    SemanticIndexCascade,
    IncrementalEvaluation,
    VersionedExperiment
)

# Users can then:
pipeline = ProgressiveRefinement(videos)
pipeline.add_stage('frames', FrameIterator)
pipeline.add_stage('detections', yolox)
pipeline.add_stage('alerts', alert_condition)
```

This would make the implicit explicit and accelerate adoption!

🍪 *Patterns are like cookie recipes - once you know them, you can bake infinite variations!*