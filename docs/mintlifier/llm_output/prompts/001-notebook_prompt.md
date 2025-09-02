# Notebook Processing Prompt v001
*For creating JSONLD lesson files from Jupyter notebooks*

## MISSION
Transform a Jupyter notebook into a structured JSONLD file that helps LLMs understand Pixeltable deeply.

## STRUCTURE REQUIREMENTS

```json
{
  "@context": "https://pixeltable.com/learn",
  "@type": "Tutorial",
  "@id": "notebook-name",
  "github_url": "CRITICAL - link to actual notebook",
  "title": "From notebook",
  "objective": "What will the user be able to do after this?",
  "difficulty": "beginner|intermediate|advanced",
  "categories": ["list", "of", "topics"],
  "prerequisites": ["what should they know first?"],
  "imports_required": ["all imports used"],
  "performance_notes": {
    "typical_runtime": "how long for the whole notebook?",
    "resource_requirements": "GPU? RAM? Disk?"
  },
  "key_learnings": ["conceptual takeaways"],
  "steps": [...],
  "patterns": [...],
  "common_errors": [...],
  "test_questions": [...],
  "cookies": "🍪" 
}
```

## FOR EACH STEP

```json
{
  "number": n,
  "section_title": "From markdown headers if present",
  "intent": "What are we trying to accomplish?",
  "code": "EXACT code from cell",
  "imports_used": ["track cumulative imports"],
  "explanation": "Why this matters",
  "actual_output": "REAL output or representative sample",
  "output_type": "table|json|image|text|number",
  "learns": ["new concepts introduced"],
  "gotchas": ["weird things that might trip you up"],
  "performance": "execution time if notable",
  "alternatives": "other ways to do this?"
}
```

## PATTERNS TO EXTRACT

```json
{
  "name": "pattern_name",
  "description": "when to use this",
  "code_template": "generalized version",
  "variations": ["different ways to do it"],
  "reusable": true/false
}
```

## COMMON ERRORS

```json
{
  "error_type": "what error message",
  "cause": "why it happens",
  "solution": "how to fix",
  "example": "code that causes it"
}
```

## EXTRACTION RULES

1. **BE LITERAL**: Copy code EXACTLY, including comments
2. **CAPTURE OUTPUTS**: Include actual outputs (truncate if huge)
3. **NOTE THE WEIRD**: Anything surprising or non-obvious
4. **TRACK STATE**: What variables/tables exist at each step
5. **IDENTIFY DEPENDENCIES**: What needs to happen before this works
6. **MEASURE TIME**: Note anything that takes >5 seconds
7. **LINK EVERYTHING**: GitHub URLs, doc links, related notebooks

## QUESTIONS TO ASK YOURSELF

- [ ] Would an LLM know what packages to install?
- [ ] Can an LLM reproduce this exactly?
- [ ] Are the gotchas clear?
- [ ] Is the progression logical?
- [ ] Did I capture the "aha!" moments?
- [ ] Would this help debug common errors?
- [ ] Is it grep-friendly?
- [ ] Did I include cookies? 🍪

## OUTPUT VALIDATION

After creating, test with:
```bash
# Can I find specific concepts?
grep -A 5 "computed_column" file.jsonld

# Can I extract all gotchas?
jq '.steps[].gotchas' file.jsonld

# Can I get all code for a pattern?
jq '.patterns[] | select(.name == "basic_pipeline")' file.jsonld
```

## PRIORITY ORDER
1. Correctness (is it right?)
2. Completeness (is everything there?)
3. Queryability (can I find things?)
4. Clarity (is it understandable?)
5. Cookies (are there cookies? 🍪)