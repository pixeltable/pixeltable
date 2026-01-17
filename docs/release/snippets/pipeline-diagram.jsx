export const PipelineDiagram = () => {
  const linkClass = "hover:text-blue-600 dark:hover:text-blue-400 hover:underline transition-colors"
  
  return (
    <div className="w-full bg-white dark:bg-neutral-900 p-3 rounded-lg border border-neutral-200 dark:border-neutral-700 not-prose overflow-x-auto">
      <div className="flex items-stretch gap-1.5 min-w-[680px]">
        {/* STORAGE Section */}
        <div className="w-[130px] flex-shrink-0 border border-neutral-300 dark:border-neutral-600 p-2.5 bg-white dark:bg-neutral-800 rounded">
          <h2 className="font-mono text-[11px] tracking-widest mb-1.5 uppercase text-neutral-900 dark:text-neutral-100 font-semibold">
            <a href="/tutorials/tables-and-data-operations" className={linkClass}>STORAGE</a>
          </h2>
          <h3 className="font-mono text-[9px] mb-0.5 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
            <a href="/platform/type-system" className={linkClass}>Data Tables</a>
          </h3>
          <div className="border-b border-neutral-200 dark:border-neutral-600 mb-1.5" />
          <ul className="space-y-0.5 text-[12px] font-mono text-neutral-900 dark:text-neutral-100">
            <li><a href="/platform/type-system" className={linkClass}>Structured</a></li>
            <li><a href="/platform/type-system" className={linkClass}>Video/Image</a></li>
            <li><a href="/platform/type-system" className={linkClass}>Audio/Doc</a></li>
            <li><a href="/platform/type-system" className={linkClass}>JSON/Text</a></li>
          </ul>
        </div>

        {/* Arrow 1 */}
        <div className="flex items-center flex-shrink-0 text-neutral-400 dark:text-neutral-500 px-0.5">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        </div>

        {/* ORCHESTRATION Section */}
        <div className="flex-1 min-w-[320px] border border-neutral-300 dark:border-neutral-600 p-2.5 bg-white dark:bg-neutral-800 rounded">
          <h2 className="font-mono text-[11px] tracking-widest mb-1.5 uppercase text-neutral-900 dark:text-neutral-100 font-semibold">
            <a href="/tutorials/computed-columns" className={linkClass}>ORCHESTRATION</a>
          </h2>

          <div className="flex gap-3">
            {/* Pre-process */}
            <div className="flex-1">
              <h3 className="font-mono text-[9px] mb-0.5 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Pre-process
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-1" />
              <ul className="space-y-0.5 text-[12px] font-mono text-neutral-900 dark:text-neutral-100">
                <li><a href="/tutorials/computed-columns" className={linkClass}>Transform</a></li>
                <li><a href="/platform/iterators" className={linkClass}>Chunk/Split</a></li>
                <li><a href="/platform/udfs-in-pixeltable" className={linkClass}>Validate</a></li>
              </ul>
            </div>

            {/* Generate */}
            <div className="flex-1">
              <h3 className="font-mono text-[9px] mb-0.5 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Generate
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-1" />
              <ul className="space-y-0.5 text-[12px] font-mono text-neutral-900 dark:text-neutral-100 whitespace-nowrap">
                <li><a href="/integrations/frameworks" className={linkClass}>LLM calls</a></li>
                <li><a href="/integrations/frameworks" className={linkClass}>Local inference</a></li>
                <li><a href="/platform/embedding-indexes" className={linkClass}>Embeddings</a></li>
              </ul>
            </div>

            {/* Post-process */}
            <div className="flex-1">
              <h3 className="font-mono text-[9px] mb-0.5 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Post-process
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-1" />
              <ul className="space-y-0.5 text-[12px] font-mono text-neutral-900 dark:text-neutral-100">
                <li><a href="/tutorials/queries-and-expressions" className={linkClass}>Aggregate</a></li>
                <li><a href="/tutorials/computed-columns" className={linkClass}>Extract</a></li>
                <li><a href="/platform/embedding-indexes" className={linkClass}>Index</a></li>
              </ul>
            </div>
          </div>
        </div>

        {/* Arrow 2 */}
        <div className="flex items-center flex-shrink-0 text-neutral-400 dark:text-neutral-500 px-0.5">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        </div>

        {/* RETRIEVAL Section */}
        <div className="w-[130px] flex-shrink-0 border border-neutral-300 dark:border-neutral-600 p-2.5 bg-white dark:bg-neutral-800 rounded">
          <h2 className="font-mono text-[11px] tracking-widest mb-1.5 uppercase text-neutral-900 dark:text-neutral-100 font-semibold">
            <a href="/tutorials/queries-and-expressions" className={linkClass}>RETRIEVAL</a>
          </h2>
          <h3 className="font-mono text-[9px] mb-0.5 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
            Query & Serve
          </h3>
          <div className="border-b border-neutral-200 dark:border-neutral-600 mb-1.5" />
          <ul className="space-y-0.5 text-[12px] font-mono text-neutral-900 dark:text-neutral-100">
            <li><a href="/tutorials/queries-and-expressions" className={linkClass}>SQL-like</a></li>
            <li><a href="/platform/embedding-indexes" className={linkClass}>Similarity</a></li>
            <li><a href="/use-cases/ml-data-wrangling" className={linkClass}>Export (ML)</a></li>
            <li><a href="/platform/data-sharing" className={linkClass}>Share/Publish</a></li>
            <li><a href="/use-cases/agents-mcp" className={linkClass}>Tools/MCP</a></li>
            <li><a href="/use-cases/ai-applications" className={linkClass}>FastAPI</a></li>
          </ul>
        </div>
      </div>
    </div>
  )
}
