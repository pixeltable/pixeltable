"""
Pixeltable Documentation Learning Pipeline
Using Pixeltable to process, learn from, and improve its own documentation
"""

import pixeltable as pxt
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions import openai, anthropic, huggingface
from typing import Dict, List, Any
import json
from datetime import datetime

# ========================================
# PHASE 1: Setup Tables and Schema
# ========================================

def setup_documentation_pipeline():
    """Initialize the complete documentation learning system"""
    
    # Clean slate for demo (the universal pattern!)
    pxt.drop_dir('doc_learning', force=True)
    pxt.create_dir('doc_learning')
    
    # ----------------------------------------
    # 1. Source Notebooks Table
    # ----------------------------------------
    notebooks = pxt.create_table('doc_learning.notebooks', {
        'notebook_path': pxt.String,
        'notebook_content': pxt.Document,  # Store as document
        'category': pxt.String,  # fundamentals, use-cases, integrations
        'difficulty': pxt.String,  # beginner, intermediate, advanced
        'added_at': pxt.Timestamp,
        'version': pxt.Int  # Track notebook versions
    })
    
    # ----------------------------------------
    # 2. Processed Lessons Table (JSONLD output)
    # ----------------------------------------
    lessons = pxt.create_table('doc_learning.lessons', {
        'notebook_id': pxt.String,
        'lesson_jsonld': pxt.Json,  # Full JSONLD structure
        'patterns_extracted': pxt.Json,  # List of patterns found
        'quality_score': pxt.Float,
        'processing_time': pxt.Float,
        'prompt_version': pxt.String,
        'processed_at': pxt.Timestamp
    })
    
    # ----------------------------------------
    # 3. Patterns Master Table
    # ----------------------------------------
    patterns = pxt.create_table('doc_learning.patterns', {
        'pattern_name': pxt.String,
        'pattern_code': pxt.String,
        'description': pxt.String,
        'frequency': pxt.Int,
        'confidence': pxt.String,  # novel, emerging, established, saturated
        'first_seen_in': pxt.String,
        'category': pxt.String,
        'production_ready': pxt.Bool,
        'dependencies': pxt.Json,  # List of prerequisite patterns
        'enables': pxt.Json  # List of patterns this enables
    })
    
    # ----------------------------------------
    # 4. Prompts Evolution Table
    # ----------------------------------------
    prompts = pxt.create_table('doc_learning.prompts', {
        'version': pxt.String,
        'prompt_text': pxt.String,
        'parent_version': pxt.String,
        'improvements': pxt.Json,
        'performance_metrics': pxt.Json,
        'created_at': pxt.Timestamp
    })
    
    # ----------------------------------------
    # 5. Quality Metrics Table
    # ----------------------------------------
    metrics = pxt.create_table('doc_learning.metrics', {
        'lesson_id': pxt.String,
        'code_accuracy': pxt.Float,
        'output_completeness': pxt.Float,
        'pattern_recognition': pxt.Float,
        'cross_references': pxt.Int,
        'production_tips': pxt.Int,
        'cookie_quality': pxt.Float  # 🍪 Essential metric!
    })
    
    return notebooks, lessons, patterns, prompts, metrics


# ========================================
# PHASE 2: Processing Pipeline (Progressive Refinement Pattern)
# ========================================

def setup_processing_pipeline(notebooks, lessons, patterns):
    """Implement the progressive refinement pipeline for documentation"""
    
    # ----------------------------------------
    # Stage 1: Notebook Chunking
    # ----------------------------------------
    # Create view for notebook cells (one-to-many expansion)
    cells = pxt.create_view(
        'doc_learning.notebook_cells',
        notebooks,
        iterator=DocumentSplitter.create(
            document=notebooks.notebook_content,
            separators='cell',  # Custom separator for notebook cells
            metadata='cell_type,execution_count'
        )
    )
    
    # ----------------------------------------
    # Stage 2: Cell Classification
    # ----------------------------------------
    @pxt.udf
    def classify_cell(cell_content: str, cell_type: str) -> str:
        """Classify cell purpose: setup, core_logic, visualization, etc."""
        if 'import' in cell_content or 'pip install' in cell_content:
            return 'setup'
        elif 'pxt.create_table' in cell_content:
            return 'table_creation'
        elif 'add_computed_column' in cell_content:
            return 'computed_column'
        elif '.show()' in cell_content or '.collect()' in cell_content:
            return 'query'
        else:
            return 'other'
    
    cells.add_computed_column(
        cell_classification=classify_cell(cells.text, cells.cell_type)
    )
    
    # ----------------------------------------
    # Stage 3: Pattern Extraction (Multi-Model Consensus Pattern)
    # ----------------------------------------
    
    # Use multiple LLMs to extract patterns
    cells.add_computed_column(
        patterns_gpt4=openai.chat_completions(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': 'Extract Pixeltable patterns from this code'},
                {'role': 'user', 'content': cells.text}
            ]
        )
    )
    
    cells.add_computed_column(
        patterns_claude=anthropic.chat_completions(
            model='claude-3-opus',
            messages=[
                {'role': 'system', 'content': 'Extract Pixeltable patterns from this code'},
                {'role': 'user', 'content': cells.text}
            ]
        )
    )
    
    # Consensus pattern extraction
    @pxt.udf
    def consensus_patterns(gpt4_patterns: Dict, claude_patterns: Dict) -> List[str]:
        """Combine patterns from multiple models"""
        # Extract pattern lists from both
        patterns_1 = json.loads(gpt4_patterns.get('content', '[]'))
        patterns_2 = json.loads(claude_patterns.get('content', '[]'))
        
        # Find consensus (patterns identified by both)
        consensus = list(set(patterns_1) & set(patterns_2))
        
        # High confidence patterns appear in both
        return consensus
    
    cells.add_computed_column(
        consensus_patterns=consensus_patterns(
            cells.patterns_gpt4,
            cells.patterns_claude
        )
    )
    
    # ----------------------------------------
    # Stage 4: Lesson Generation
    # ----------------------------------------
    
    @pxt.udf
    def generate_lesson(notebook_path: str, prompt_version: str) -> Dict:
        """Generate complete JSONLD lesson from notebook"""
        # This would use the full v002 prompt methodology
        # For now, returning structure
        return {
            '@context': 'https://pixeltable.com/learn',
            '@type': 'Tutorial',
            '@id': notebook_path.split('/')[-1].replace('.ipynb', ''),
            'patterns': [],  # Would be filled by aggregation
            'steps': [],  # Would be filled by processing
            'cookies': '🍪'  # Always!
        }
    
    # Add to lessons table as computed column on notebooks
    notebooks.add_computed_column(
        lesson=generate_lesson(notebooks.notebook_path, 'v002')
    )
    
    return cells


# ========================================
# PHASE 3: Pattern Recognition System (Semantic Index Cascade)
# ========================================

def setup_pattern_recognition(patterns, lessons):
    """Build multi-level pattern recognition system"""
    
    # ----------------------------------------
    # Create pattern embeddings at different granularities
    # ----------------------------------------
    
    # Code-level patterns (fine-grained)
    patterns.add_computed_column(
        code_embedding=huggingface.sentence_transformer(
            patterns.pattern_code,
            model_id='microsoft/codebert-base'
        )
    )
    
    # Description-level patterns (semantic)
    patterns.add_computed_column(
        semantic_embedding=openai.embeddings(
            patterns.description,
            model='text-embedding-3-small'
        )
    )
    
    # Create indexes for similarity search
    patterns.add_embedding_index(
        'pattern_code',
        embedding=huggingface.sentence_transformer.using(
            model_id='microsoft/codebert-base'
        )
    )
    
    patterns.add_embedding_index(
        'description',
        embedding=openai.embeddings.using(
            model='text-embedding-3-small'
        )
    )
    
    # ----------------------------------------
    # Pattern Frequency Analysis
    # ----------------------------------------
    
    @pxt.udf
    def calculate_pattern_confidence(frequency: int, total_notebooks: int) -> str:
        """Determine pattern maturity based on frequency"""
        ratio = frequency / total_notebooks
        if ratio < 0.1:
            return 'novel'
        elif ratio < 0.3:
            return 'emerging'
        elif ratio < 0.6:
            return 'established'
        else:
            return 'saturated'
    
    patterns.add_computed_column(
        confidence=calculate_pattern_confidence(
            patterns.frequency,
            pxt.functions.count(patterns.pattern_name)  # Total notebooks
        )
    )


# ========================================
# PHASE 4: Self-Improvement Loop (Incremental Evaluation)
# ========================================

def setup_improvement_loop(prompts, lessons, metrics):
    """Implement self-improving prompt system"""
    
    # ----------------------------------------
    # Evaluate prompt performance
    # ----------------------------------------
    
    @pxt.udf
    def evaluate_prompt_performance(lesson: Dict) -> Dict:
        """Score prompt performance on various metrics"""
        return {
            'pattern_discovery_rate': len(lesson.get('patterns', [])) / 10,
            'step_completeness': len(lesson.get('steps', [])) / 5,
            'has_cookies': 1.0 if '🍪' in str(lesson) else 0.0,
            'overall_quality': 0.85  # Would be computed
        }
    
    lessons.add_computed_column(
        prompt_performance=evaluate_prompt_performance(lessons.lesson_jsonld)
    )
    
    # ----------------------------------------
    # Generate improved prompts
    # ----------------------------------------
    
    @pxt.udf
    def generate_improved_prompt(
        current_prompt: str,
        performance_metrics: Dict,
        feedback: str
    ) -> str:
        """Generate next version of prompt based on performance"""
        # This would use LLM to improve the prompt
        # For now, return enhanced version
        improvements = []
        
        if performance_metrics['pattern_discovery_rate'] < 0.7:
            improvements.append('Add more pattern recognition rules')
        
        if performance_metrics['step_completeness'] < 0.8:
            improvements.append('Improve step extraction logic')
        
        if performance_metrics['has_cookies'] < 1.0:
            improvements.append('ALWAYS ADD COOKIES! 🍪')
        
        return current_prompt + '\n\n## Improvements\n' + '\n'.join(improvements)
    
    # Create view for prompt evolution
    prompt_evolution = pxt.create_view(
        'doc_learning.prompt_evolution',
        prompts
    )
    
    prompt_evolution.add_computed_column(
        next_version=generate_improved_prompt(
            prompt_evolution.prompt_text,
            pxt.functions.aggregate.mean(lessons.prompt_performance),
            'Improve pattern recognition'  # Feedback
        )
    )


# ========================================
# PHASE 5: Query Interface
# ========================================

def setup_query_interface():
    """Create queryable interface for documentation"""
    
    # ----------------------------------------
    # Common Queries as Views
    # ----------------------------------------
    
    # View: All production-ready patterns
    production_patterns = pxt.create_view(
        'doc_learning.production_patterns',
        pxt.get_table('doc_learning.patterns')
    ).where(
        pxt.get_table('doc_learning.patterns').production_ready == True
    )
    
    # View: Recently discovered patterns
    recent_patterns = pxt.create_view(
        'doc_learning.recent_patterns',
        pxt.get_table('doc_learning.patterns')
    ).where(
        pxt.get_table('doc_learning.patterns').confidence == 'novel'
    ).order_by(
        pxt.get_table('doc_learning.patterns').first_seen_in,
        asc=False
    ).limit(10)
    
    # ----------------------------------------
    # Search Functions
    # ----------------------------------------
    
    @pxt.udf
    def find_pattern_for_task(task_description: str) -> List[Dict]:
        """Find relevant patterns for a given task"""
        patterns_table = pxt.get_table('doc_learning.patterns')
        
        # Search by semantic similarity
        similar_patterns = patterns_table.order_by(
            patterns_table.description.similarity(task_description),
            asc=False
        ).limit(5).collect()
        
        return similar_patterns
    
    @pxt.udf
    def generate_code_for_use_case(use_case: str) -> str:
        """Generate Pixeltable code for a use case using learned patterns"""
        # Find relevant patterns
        patterns = find_pattern_for_task(use_case)
        
        # Combine patterns into solution
        code_template = []
        for pattern in patterns:
            code_template.append(pattern['pattern_code'])
        
        return '\n\n'.join(code_template)
    
    return production_patterns, recent_patterns


# ========================================
# MAIN EXECUTION
# ========================================

def run_documentation_pipeline():
    """Execute the complete documentation learning pipeline"""
    
    print("🚀 Setting up Pixeltable Documentation Pipeline...")
    
    # Phase 1: Initialize tables
    notebooks, lessons, patterns, prompts, metrics = setup_documentation_pipeline()
    print("✅ Tables created")
    
    # Phase 2: Setup processing pipeline
    cells = setup_processing_pipeline(notebooks, lessons, patterns)
    print("✅ Processing pipeline ready")
    
    # Phase 3: Pattern recognition
    setup_pattern_recognition(patterns, lessons)
    print("✅ Pattern recognition system initialized")
    
    # Phase 4: Self-improvement
    setup_improvement_loop(prompts, lessons, metrics)
    print("✅ Self-improvement loop activated")
    
    # Phase 5: Query interface
    production_patterns, recent_patterns = setup_query_interface()
    print("✅ Query interface ready")
    
    # ----------------------------------------
    # Load initial notebooks
    # ----------------------------------------
    
    print("\n📚 Loading notebooks...")
    
    # Insert our processed notebooks
    notebook_paths = [
        'pixeltable-basics.ipynb',
        'object-detection-in-videos.ipynb',
        'embedding-indexes.ipynb',
        'rag-operations.ipynb',
        'udfs-in-pixeltable.ipynb'
    ]
    
    for path in notebook_paths:
        notebooks.insert({
            'notebook_path': f'/docs/notebooks/{path}',
            'notebook_content': f'https://github.com/pixeltable/pixeltable/blob/main/docs/notebooks/{path}',
            'category': 'fundamentals' if 'basics' in path else 'use-cases',
            'difficulty': 'beginner' if 'basics' in path else 'intermediate',
            'added_at': datetime.now(),
            'version': 1
        })
    
    print(f"✅ Loaded {len(notebook_paths)} notebooks")
    
    # ----------------------------------------
    # Show results
    # ----------------------------------------
    
    print("\n📊 Pipeline Statistics:")
    print(f"Notebooks processed: {notebooks.count()}")
    print(f"Lessons generated: {lessons.count()}")
    print(f"Patterns discovered: {patterns.count()}")
    print(f"Current prompt version: v002")
    
    print("\n🔍 Sample Pattern Search:")
    print("Query: 'How to process video frames?'")
    # This would actually run the similarity search
    
    print("\n🍪 Documentation pipeline is learning and improving!")
    
    return notebooks, lessons, patterns, prompts, metrics


# ========================================
# BONUS: Meta-Documentation Generation
# ========================================

@pxt.udf
def generate_documentation_about_documentation() -> str:
    """
    The ultimate recursion: Use the documentation pipeline
    to document itself!
    """
    return """
    # How Pixeltable Documents Itself
    
    This pipeline demonstrates the recursive beauty of Pixeltable:
    
    1. **Notebooks** are treated as Documents
    2. **Cells** are extracted via DocumentSplitter
    3. **Patterns** are recognized via computed columns
    4. **Lessons** are generated via LLM integration
    5. **Quality** improves via evaluation loops
    6. **Knowledge** accumulates in persistent tables
    
    The same patterns we discovered ARE the patterns we use:
    - Progressive Refinement (notebook → cells → patterns → lessons)
    - Multi-Model Consensus (GPT-4 + Claude for pattern extraction)
    - Semantic Index Cascade (code + description embeddings)
    - Incremental Evaluation (continuous prompt improvement)
    - Versioned Experiments (prompt versions)
    
    🍪 Even the cookies are computed columns!
    """


if __name__ == "__main__":
    # Run the complete pipeline
    notebooks, lessons, patterns, prompts, metrics = run_documentation_pipeline()
    
    # The beautiful part: This entire pipeline is incremental!
    # Add a new notebook → Automatically processed
    # Patterns discovered → Automatically catalogued
    # Prompt improves → Automatically versioned
    
    print("\n✨ Pixeltable is now documenting itself!")
    print("🍪 With cookies at every step!")