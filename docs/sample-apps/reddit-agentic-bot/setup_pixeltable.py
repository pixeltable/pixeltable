# Standard library imports
import uuid

# Third-party library imports
from dotenv import load_dotenv

# Import centralized configuration **before** Pixeltable/UDF imports
import config

# Pixeltable core imports
import pixeltable as pxt
import pixeltable.iterators as pxt_iterators  # Explicit import for clarity

# Pixeltable function imports - organized by category
from pixeltable.functions.anthropic import invoke_tools, messages
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions import string as pxt_str

# Custom function imports (UDFs and Queries for LLM steps)
import functions

# Load environment variables
load_dotenv()

print(f"--- Initializing Pixeltable Environment for '{config.BASE_DIR}' ---")

# Clean slate: Always attempt to drop the directory
pxt.drop_dir(
    config.BASE_DIR, force=True
)  # Removes the specified directory and all contents from Pixeltable's storage backend.
pxt.create_dir(
    config.BASE_DIR, if_exists="ignore"
)  # Creates a logical directory within Pixeltable for organization.

# === DOCUMENT PROCESSING ===
docs_table_path = f"{config.BASE_DIR}.documents"
docs_schema = {
    "doc_id": pxt.String,           # Standard string type
    "source_uri": pxt.String,       # Explicitly store original source URL/path
    "related_url": pxt.String, # Optional URL associated with the source (e.g., for PDFs)
    "doc": pxt.Document,            # Pixeltable type for content + internal metadata
}
documents = pxt.create_table(
    docs_table_path,
    docs_schema,  # Define column names and types.
    if_exists="ignore",
)

# Optional: Ingest initial source data.
new_data = []
# Iterate over the list of dictionaries in SOURCE_DATA
for item in config.SOURCE_DATA:
    src = item.get("source")
    related = item.get("related_url") # Get optional related URL
    if not src: # Basic check
        print(f"Warning: Skipping item in SOURCE_DATA with missing 'source': {item}")
        continue

    # Map source URLs/paths to the table schema
    new_data.append({
        "doc_id": str(uuid.uuid4()),
        "source_uri": src,         # Store original source
        "related_url": related,    # Store the related URL (can be None)
        "doc": src                 # Let Pixeltable process the content from 'source'
    })
if new_data:
    # Bulk insert data.
    documents.insert(new_data, on_error="ignore")
else:
    print("No sources found in config.SOURCE_DATA to insert.")

# === CHUNKING AND EMBEDDING ===
chunks_view_path = f"{config.BASE_DIR}.doc_chunks"
# Create a view: a virtual table derived from another table/view.
# The view will automatically include the new source_uri column from the base table.
doc_chunks = pxt.create_view(
    chunks_view_path,
    documents,  # Base table for the view.
    # Use an Iterator to generate multiple output rows (chunks).
    iterator=pxt_iterators.DocumentSplitter.create(
        document=documents.doc,
        separators='token_limit',
        limit=250,
        overlap=0,
        metadata="title,heading,sourceline"
    ),
    # Schema is automatically inferred (+ heading, + source_uri).
    if_exists="ignore",
)

# Add an embedding index to accelerate similarity searches.
doc_chunks.add_embedding_index(
    column="text",
    string_embed=sentence_transformer.using(model_id=config.EMBEDDING_MODEL_ID),
    if_exists="ignore",
)


# === QUERY FUNCTION DEFINITION ===
@pxt.query
def search_document_chunks(query_text: str):
    """Search indexed document chunks for relevant context, filtering by similarity."""
    sim = doc_chunks.text.similarity(query_text)
    results = (
        doc_chunks.where((sim >= config.MIN_SIMILARITY_THRESHOLD) & (pxt_str.len(doc_chunks.text) > 50))
        .order_by(sim, asc=False)
        .limit(config.NUM_CONTEXT_CHUNKS)
        .select(
            text=doc_chunks.text,
            # title=doc_chunks.title, # Removing potentially unreliable metadata
            # heading=doc_chunks.heading,
            # sourceline=doc_chunks.sourceline,
            source_uri=doc_chunks.source_uri,
            related_url=doc_chunks.related_url, # Select the new column
            similarity=sim,
        )
    )
    return results


# === TOOL REGISTRATION ===
tools = pxt.tools(functions.run_duckduckgo_search, functions.fetch_financial_data)

# === QUESTIONS WORKFLOW TABLE ===
questions_table_path = f"{config.BASE_DIR}.questions"
q_schema = {
    "reddit_id": pxt.String,
    "subreddit": pxt.String,
    "author": pxt.String,
    "question_text": pxt.String,
    "timestamp": pxt.Timestamp,
    "status": pxt.String,
}
questions = pxt.create_table(questions_table_path, q_schema, if_exists="ignore")

# === DECLARATIVE WORKFLOW WITH COMPUTED COLUMNS ===

# a. Retrieve Context (Now includes source_uri)
questions.add_computed_column(
    retrieved_context=search_document_chunks(questions.question_text),
    if_exists="replace",
)

# --- Branch 1: Tool Usage ---

# b. Format Initial Prompt (Needs source_uri from retrieved_context)
questions.add_computed_column(
    initial_prompt=functions.format_initial_prompt(
        questions.question_text, questions.retrieved_context
    ),
    if_exists="replace",
)

# c. Call LLM 1 (Tool Selection)
questions.add_computed_column(
    llm_response_1=messages(
        model=config.LLM_MODEL_ID,
        system=config.SYSTEM_MESSAGE,
        messages=questions.initial_prompt,
        tools=tools,
    ),
    if_exists="replace",
)

# d. Invoke Tools
questions.add_computed_column(
    tool_output=invoke_tools(tools, questions.llm_response_1),
    if_exists="replace"
)

# --- Branch 2: General Knowledge ---

# e. Call LLM 2 (General Knowledge Answer)
questions.add_computed_column(
    llm_general_response=messages(
        model=config.LLM_MODEL_ID,
        system=config.GENERAL_KNOWLEDGE_SYSTEM_MESSAGE,
        messages=[{"role": "user", "content": questions.question_text}],
    ),
    if_exists="replace",
)

# --- Final Synthesis ---

# f. Format Synthesis Prompt Input (Needs source_uri from retrieved_context)
questions.add_computed_column(
    synthesis_prompt_messages=functions.format_synthesis_messages(
        questions.question_text,
        questions.retrieved_context,
        questions.tool_output,
        questions.llm_general_response,
    ),
    if_exists="replace",
)

# g. Call LLM 3 (Synthesis)
questions.add_computed_column(
    llm_synthesis_response=messages(
        model=config.LLM_MODEL_ID,
        system=config.SYNTHESIS_SYSTEM_MESSAGE,
        messages=questions.synthesis_prompt_messages,
    ),
    if_exists="replace",
)

# h. Extract Final Answer from Synthesis
questions.add_computed_column(
    final_answer=questions.llm_synthesis_response.content[0].text,
    if_exists="replace",
)

print("\n✅ Data store and computation setup complete.")
print("You can now run the 'reddit_bot.py' script.")
