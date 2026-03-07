const ArrowDown = () => (
  <div className="flex justify-center py-2 text-stone-300 dark:text-stone-600">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 5v14M5 13l7 7 7-7" />
    </svg>
  </div>
)

const ArrowRight = () => (
  <div className="flex items-center text-stone-300 dark:text-stone-600 flex-shrink-0">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M5 12h14M13 5l7 7-7 7" />
    </svg>
  </div>
)

const TraditionalBox = ({ label }) => (
  <div className="rounded border border-stone-300 dark:border-stone-600 bg-stone-50 dark:bg-stone-800 px-2.5 py-1.5 text-[11px] font-medium text-stone-600 dark:text-stone-300 text-center whitespace-nowrap">
    {label}
  </div>
)

const PxtBox = ({ label, href, sub }) => (
  <a href={href} className="block rounded border border-[#022A59]/20 dark:border-[#F1AE03]/20 bg-[#022A59]/5 dark:bg-[#F1AE03]/5 px-2.5 py-1.5 text-center no-underline hover:border-[#022A59]/40 dark:hover:border-[#F1AE03]/40 transition-colors">
    <span className="text-[11px] font-semibold text-[#022A59] dark:text-[#F1AE03] block">{label}</span>
    {sub && <span className="text-[9px] text-stone-500 dark:text-stone-400 block mt-0.5">{sub}</span>}
  </a>
)

export const PipelineDiagram = () => {
  return (
    <div className="w-full not-prose overflow-x-auto">
      <div className="min-w-[700px] space-y-3">

        {/* WITHOUT PIXELTABLE */}
        <div className="rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-900 p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-[10px] font-bold uppercase tracking-widest text-stone-400 dark:text-stone-500">Without Pixeltable</span>
            <span className="text-[10px] text-stone-400 dark:text-stone-500">— 6+ services to deploy and maintain</span>
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            <TraditionalBox label="S3 / GCS" />
            <ArrowRight />
            <TraditionalBox label="Airflow / Prefect" />
            <ArrowRight />
            <div className="flex gap-1.5">
              <TraditionalBox label="PostgreSQL" />
              <TraditionalBox label="Pinecone / Weaviate" />
            </div>
            <ArrowRight />
            <div className="flex gap-1.5">
              <TraditionalBox label="FFmpeg" />
              <TraditionalBox label="spaCy" />
              <TraditionalBox label="LangChain" />
            </div>
            <ArrowRight />
            <TraditionalBox label="Glue Code" />
          </div>
        </div>

        <ArrowDown />

        {/* WITH PIXELTABLE */}
        <div className="rounded-lg border-2 border-[#022A59]/30 dark:border-[#F1AE03]/30 bg-white dark:bg-stone-900 p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-5 h-5 rounded bg-[#022A59] dark:bg-[#F1AE03] flex items-center justify-center flex-shrink-0">
              <svg className="w-3 h-3 text-white dark:text-stone-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <span className="text-[10px] font-bold uppercase tracking-widest text-[#022A59] dark:text-[#F1AE03]">With Pixeltable</span>
            <span className="text-[10px] text-stone-400 dark:text-stone-500">— 1 Python import</span>
          </div>

          <div className="flex items-stretch gap-2">
            {/* Store */}
            <div className="flex-1 space-y-1.5">
              <p className="text-[9px] uppercase tracking-widest text-stone-400 dark:text-stone-500 font-semibold mb-1.5">Store</p>
              <PxtBox label="pxt.create_table()" href="/tutorials/tables-and-data-operations" sub="replaces PostgreSQL + S3" />
              <PxtBox label="pxt.Image / Video / Audio" href="/platform/type-system" sub="replaces blob management" />
            </div>

            <ArrowRight />

            {/* Orchestrate */}
            <div className="flex-1 space-y-1.5">
              <p className="text-[9px] uppercase tracking-widest text-stone-400 dark:text-stone-500 font-semibold mb-1.5">Orchestrate</p>
              <PxtBox label="add_computed_column()" href="/tutorials/computed-columns" sub="replaces Airflow / ETL" />
              <PxtBox label="frame_iterator() / document_splitter()" href="/platform/iterators" sub="abstracts FFmpeg, spaCy" />
              <PxtBox label="openai / anthropic / huggingface" href="/integrations/frameworks" sub="built-in rate limiting" />
            </div>

            <ArrowRight />

            {/* Index */}
            <div className="flex-1 space-y-1.5">
              <p className="text-[9px] uppercase tracking-widest text-stone-400 dark:text-stone-500 font-semibold mb-1.5">Index</p>
              <PxtBox label="add_embedding_index()" href="/platform/embedding-indexes" sub="replaces Pinecone / Weaviate" />
              <PxtBox label="sentence_transformer.using()" href="/sdk/latest/huggingface" sub="abstracts HuggingFace" />
            </div>

            <ArrowRight />

            {/* Retrieve */}
            <div className="flex-1 space-y-1.5">
              <p className="text-[9px] uppercase tracking-widest text-stone-400 dark:text-stone-500 font-semibold mb-1.5">Retrieve</p>
              <PxtBox label=".similarity() + @pxt.query" href="/platform/embedding-indexes" sub="replaces LangChain RAG" />
              <PxtBox label="pxt.tools() + invoke_tools()" href="/howto/cookbooks/agents/llm-tool-calling" sub="agents & MCP" />
              <PxtBox label=".collect() / export" href="/tutorials/queries-and-expressions" sub="SQL-like queries, ML export" />
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
