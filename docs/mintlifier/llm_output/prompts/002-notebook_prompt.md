# Notebook Processing Prompt v002
*Enhanced with pattern recognition and cross-reference capabilities*

## MISSION
Transform Jupyter notebooks into deeply structured JSONLD files that help LLMs understand Pixeltable patterns, relationships, and production usage.

## CORE STRUCTURE

```json
{
  "@context": "https://pixeltable.com/learn",
  "@type": "Tutorial",
  "@id": "notebook-name",
  "github_url": "https://github.com/pixeltable/pixeltable/blob/release/docs/notebooks/...",
  "title": "From notebook",
  "objective": "Specific, measurable outcome",
  "difficulty": "beginner|intermediate|advanced",
  "categories": ["max 5-6 core concepts"],
  "prerequisites": ["notebook-ids that should be completed first"],
  "imports_required": ["all imports with full paths"],
  "performance_notes": {
    "typical_runtime": "X-Y minutes with GPU/CPU",
    "resource_requirements": "Specific RAM, GPU, disk, network",
    "bottlenecks": ["model downloads", "embedding computation"]
  },
  "key_learnings": ["5-7 conceptual insights"],
  "relationships": {
    "builds_on": ["concept1", "concept2"],
    "enables": ["downstream_capability1"],
    "see_also": ["notebook_id#section"],
    "contrasts_with": ["alternative_approach"]
  },
  "steps": [...],
  "patterns": [...],
  "common_errors": [...],
  "test_questions": [...],
  "production_tips": [...],
  "pattern_maturity": {
    "novel_patterns": 2,
    "established_patterns": 5,
    "total_patterns": 7
  },
  "cookies": "🍪 [contextual cookie joke]"
}
```

## ENHANCED STEP STRUCTURE

```json
{
  "number": n,
  "section_title": "From markdown if present",
  "intent": "User-focused goal",
  "code": "EXACT code including comments",
  "imports_used": ["cumulative list"],
  "explanation": "Why this approach",
  "actual_output": "REAL output (truncate with [... N lines ...] if >20 lines)",
  "output_summary": "What the output means",
  "output_type": "table|json|image|text|number|none",
  "learns": ["new concepts only"],
  "reinforces": ["concepts from previous steps"],
  "gotchas": ["specific traps with solutions"],
  "performance": {
    "execution_time": "Xs or Xms",
    "scaling": "O(n) where n is...",
    "optimization": "production|demo"
  },
  "alternatives": {
    "description": "Other ways to achieve this",
    "when_to_use": "Conditions for alternative"
  },
  "state_after": {
    "tables": ["table_names"],
    "views": ["view_names"],
    "variables": ["important_vars"],
    "models_loaded": ["model_ids"]
  },
  "pattern_refs": ["pattern_name_1", "pattern_name_2"]
}
```

## ENHANCED PATTERN STRUCTURE

```json
{
  "name": "snake_case_name",
  "description": "When and why to use",
  "confidence": "high|medium|low",
  "frequency": 3,  // seen in N notebooks
  "first_seen": "notebook_id",
  "code_template": "Complete working example",
  "parameters": {
    "param1": "description and options",
    "param2": "description and options"
  },
  "variations": [
    {
      "name": "variation_name",
      "difference": "what changes",
      "code": "example"
    }
  ],
  "prerequisites": ["required_knowledge"],
  "enables": ["downstream_patterns"],
  "performance_impact": "description",
  "reusable": true,
  "production_ready": true|false
}
```

## COMMON ERROR ENHANCEMENT

```json
{
  "error_type": "Exact error message",
  "frequency": "common|occasional|rare",
  "cause": "Root cause",
  "symptoms": ["what user sees"],
  "solution": {
    "quick_fix": "Immediate solution",
    "proper_fix": "Long-term solution"
  },
  "prevention": "How to avoid",
  "example": "Code that triggers it",
  "first_seen": "notebook_id#step"
}
```

## PRODUCTION TIPS (NEW SECTION)

```json
{
  "tip": "Specific optimization",
  "impact": "Performance improvement",
  "implementation": "How to implement",
  "trade_offs": "What you sacrifice",
  "example": "Code example"
}
```

## EXTRACTION RULES v2

### Accuracy Rules
1. **EXACT CODE**: Never paraphrase or clean up code
2. **REAL OUTPUTS**: Include actual outputs (truncate smartly)
3. **PRESERVE ERRORS**: If cell has error, document it
4. **TRACK STATE**: Monitor what exists after each step

### Pattern Recognition Rules
1. **IDENTIFY REPETITION**: Mark patterns seen before
2. **TRACK FREQUENCY**: Count pattern occurrences
3. **NOTE VARIATIONS**: Document how patterns differ
4. **LINK PATTERNS**: Connect related patterns

### Relationship Rules
1. **PREREQUISITE CHAIN**: What must exist before this works
2. **DOWNSTREAM IMPACT**: What this enables
3. **CROSS-REFERENCES**: Link to related content
4. **ALTERNATIVES**: When to use different approaches

### Performance Rules
1. **MEASURE TIME**: Note execution time if >1s
2. **IDENTIFY BOTTLENECKS**: Downloads, computation, I/O
3. **SCALING FACTORS**: What affects performance
4. **OPTIMIZATION LEVEL**: Demo vs production code

### State Tracking Rules
1. **TABLES/VIEWS**: List all created/modified
2. **VARIABLES**: Track important variables
3. **MODELS**: Note downloaded/cached models
4. **DEPENDENCIES**: Track what depends on what

## QUALITY CHECKLIST v2

- [ ] GitHub URL is correct and complete?
- [ ] All code copied exactly?
- [ ] Actual outputs included (or truncated notation)?
- [ ] Patterns identified and named?
- [ ] Relationships documented?
- [ ] State tracked after each step?
- [ ] Performance noted where relevant?
- [ ] Production tips included?
- [ ] Gotchas have solutions?
- [ ] Test questions cover main concepts?
- [ ] Cookie joke is contextual? 🍪

## TRUNCATION GUIDELINES

For large outputs:
```
[First 10 lines of actual output]
[... 485 lines truncated ...]
[Last 5 lines of actual output]
```

For repetitive outputs:
```
[First 3 examples]
[... 97 similar entries ...]
```

For binary/image outputs:
```
<PIL.Image.Image mode=RGB size=1280x720>
[Visual description if relevant]
```

## PATTERN MATURITY CLASSIFICATION

- **Novel**: First time seeing this pattern
- **Emerging**: Seen 2-3 times with variations  
- **Established**: Seen 4+ times, consistent form
- **Saturated**: No new variations expected

## RELATIONSHIP TYPES

- **builds_on**: Requires this concept
- **enables**: Makes this possible
- **extends**: Adds to this pattern
- **alternatives_to**: Different approach to same goal
- **combines_with**: Works well together
- **contrasts_with**: Opposite approach

## OUTPUT VALIDATION v2

Test your JSONLD:
```bash
# Check structure
jq . notebook.jsonld > /dev/null

# Find all patterns
jq '.patterns[].name' notebook.jsonld

# Track pattern frequency
jq '.patterns[] | {name, frequency}' notebook.jsonld

# Extract relationships
jq '.relationships' notebook.jsonld

# Get production tips
jq '.production_tips[]' notebook.jsonld

# Find steps by pattern
jq '.steps[] | select(.pattern_refs[] == "pattern_name")' notebook.jsonld
```

## PRIORITY ORDER v2
1. Correctness (exact code and outputs)
2. Relationships (how things connect)
3. Patterns (reusable knowledge)
4. Performance (execution characteristics)
5. Production (optimization guidance)
6. Cookies (contextual humor) 🍪

## META NOTES
- After 5+ notebooks, patterns stabilize
- Cross-references become critical
- Performance patterns emerge
- Production tips accumulate
- Relationships form knowledge graph

Remember: We're building a knowledge base that improves with each notebook processed!