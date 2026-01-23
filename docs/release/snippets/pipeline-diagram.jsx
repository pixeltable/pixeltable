export const PipelineDiagram = () => {
  return (
    <div className="w-full not-prose overflow-x-auto">
      <div className="flex items-stretch gap-2 min-w-[640px]">
        {/* STORAGE */}
        <div className="w-[120px] flex-shrink-0 rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-900 p-3">
          <div className="flex items-center gap-1.5 mb-2">
            <div className="w-5 h-5 rounded bg-[#022A59] dark:bg-[#F1AE03] flex items-center justify-center">
              <svg className="w-3 h-3 text-white dark:text-stone-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2 3.5 4 8 4s8-2 8-4V7M4 7c0 2 3.5 4 8 4s8-2 8-4M4 7c0-2 3.5-4 8-4s8 2 8 4" />
              </svg>
            </div>
            <span className="text-xs font-semibold text-stone-800 dark:text-stone-100">Storage</span>
          </div>
          <p className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-400 mb-1">Data Tables</p>
          <ul className="space-y-0.5 text-[11px] text-stone-600 dark:text-stone-300">
            <li><a href="/platform/type-system" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Structured</a></li>
            <li><a href="/platform/type-system" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Video / Image</a></li>
            <li><a href="/platform/type-system" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Audio / Doc</a></li>
            <li><a href="/platform/type-system" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">JSON / Text</a></li>
          </ul>
        </div>

        {/* Arrow */}
        <div className="flex items-center text-stone-300 dark:text-stone-600">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        </div>

        {/* ORCHESTRATION */}
        <div className="flex-1 min-w-[280px] rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-900 p-3">
          <div className="flex items-center gap-1.5 mb-2">
            <div className="w-5 h-5 rounded bg-[#022A59] dark:bg-[#F1AE03] flex items-center justify-center">
              <svg className="w-3 h-3 text-white dark:text-stone-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <span className="text-xs font-semibold text-stone-800 dark:text-stone-100">Orchestration</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div>
              <p className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-400 mb-1">Pre-process</p>
              <ul className="space-y-0.5 text-[11px] text-stone-600 dark:text-stone-300">
                <li><a href="/tutorials/computed-columns" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Transform</a></li>
                <li><a href="/platform/iterators" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Chunk / Split</a></li>
                <li><a href="/platform/udfs-in-pixeltable" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Validate</a></li>
              </ul>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-400 mb-1">Generate</p>
              <ul className="space-y-0.5 text-[11px] text-stone-600 dark:text-stone-300">
                <li><a href="/integrations/frameworks" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">LLM calls</a></li>
                <li><a href="/integrations/frameworks" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Local inference</a></li>
                <li><a href="/platform/embedding-indexes" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Embeddings</a></li>
              </ul>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-400 mb-1">Post-process</p>
              <ul className="space-y-0.5 text-[11px] text-stone-600 dark:text-stone-300">
                <li><a href="/tutorials/queries-and-expressions" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Aggregate</a></li>
                <li><a href="/tutorials/computed-columns" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Extract</a></li>
                <li><a href="/platform/embedding-indexes" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Index</a></li>
              </ul>
            </div>
          </div>
        </div>

        {/* Arrow */}
        <div className="flex items-center text-stone-300 dark:text-stone-600">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        </div>

        {/* RETRIEVAL */}
        <div className="w-[120px] flex-shrink-0 rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-900 p-3">
          <div className="flex items-center gap-1.5 mb-2">
            <div className="w-5 h-5 rounded bg-[#022A59] dark:bg-[#F1AE03] flex items-center justify-center">
              <svg className="w-3 h-3 text-white dark:text-stone-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <span className="text-xs font-semibold text-stone-800 dark:text-stone-100">Retrieval</span>
          </div>
          <p className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-400 mb-1">Query & Serve</p>
          <ul className="space-y-0.5 text-[11px] text-stone-600 dark:text-stone-300">
            <li><a href="/tutorials/queries-and-expressions" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">SQL-like</a></li>
            <li><a href="/platform/embedding-indexes" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Similarity</a></li>
            <li><a href="/use-cases/ml-data-wrangling" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Export (ML)</a></li>
            <li><a href="/platform/data-sharing" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Share / Publish</a></li>
            <li><a href="/use-cases/agents-mcp" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">Tools / MCP</a></li>
            <li><a href="/use-cases/ai-applications" className="hover:text-[#022A59] dark:hover:text-[#F1AE03]">FastAPI</a></li>
          </ul>
        </div>
      </div>
    </div>
  )
}
