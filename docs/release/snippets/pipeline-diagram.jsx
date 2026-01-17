export const PipelineDiagram = () => {
  return (
    <div className="w-full not-prose overflow-x-auto my-4">
      <div className="flex items-stretch gap-3 min-w-[720px]">
        {/* STORAGE Section */}
        <div className="w-[150px] flex-shrink-0 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-7 h-7 rounded-lg bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center">
              <svg className="w-4 h-4 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Storage</h3>
          </div>
          <p className="text-[11px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2 font-medium">Data Tables</p>
          <ul className="space-y-1.5 text-[13px] text-gray-700 dark:text-gray-300">
            <li><a href="/platform/type-system" className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">Structured</a></li>
            <li><a href="/platform/type-system" className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">Video / Image</a></li>
            <li><a href="/platform/type-system" className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">Audio / Doc</a></li>
            <li><a href="/platform/type-system" className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">JSON / Text</a></li>
          </ul>
        </div>

        {/* Arrow 1 */}
        <div className="flex items-center flex-shrink-0 text-gray-300 dark:text-gray-600">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        </div>

        {/* ORCHESTRATION Section */}
        <div className="flex-1 min-w-[340px] rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-7 h-7 rounded-lg bg-purple-100 dark:bg-purple-900/50 flex items-center justify-center">
              <svg className="w-4 h-4 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Orchestration</h3>
          </div>

          <div className="grid grid-cols-3 gap-4">
            {/* Pre-process */}
            <div>
              <p className="text-[11px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2 font-medium">Pre-process</p>
              <ul className="space-y-1.5 text-[13px] text-gray-700 dark:text-gray-300">
                <li><a href="/tutorials/computed-columns" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">Transform</a></li>
                <li><a href="/platform/iterators" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">Chunk / Split</a></li>
                <li><a href="/platform/udfs-in-pixeltable" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">Validate</a></li>
              </ul>
            </div>

            {/* Generate */}
            <div>
              <p className="text-[11px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2 font-medium">Generate</p>
              <ul className="space-y-1.5 text-[13px] text-gray-700 dark:text-gray-300">
                <li><a href="/integrations/frameworks" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">LLM calls</a></li>
                <li><a href="/integrations/frameworks" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">Local inference</a></li>
                <li><a href="/platform/embedding-indexes" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">Embeddings</a></li>
              </ul>
            </div>

            {/* Post-process */}
            <div>
              <p className="text-[11px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2 font-medium">Post-process</p>
              <ul className="space-y-1.5 text-[13px] text-gray-700 dark:text-gray-300">
                <li><a href="/tutorials/queries-and-expressions" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">Aggregate</a></li>
                <li><a href="/tutorials/computed-columns" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">Extract</a></li>
                <li><a href="/platform/embedding-indexes" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">Index</a></li>
              </ul>
            </div>
          </div>
        </div>

        {/* Arrow 2 */}
        <div className="flex items-center flex-shrink-0 text-gray-300 dark:text-gray-600">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        </div>

        {/* RETRIEVAL Section */}
        <div className="w-[150px] flex-shrink-0 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-7 h-7 rounded-lg bg-emerald-100 dark:bg-emerald-900/50 flex items-center justify-center">
              <svg className="w-4 h-4 text-emerald-600 dark:text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Retrieval</h3>
          </div>
          <p className="text-[11px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2 font-medium">Query & Serve</p>
          <ul className="space-y-1.5 text-[13px] text-gray-700 dark:text-gray-300">
            <li><a href="/tutorials/queries-and-expressions" className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">SQL-like</a></li>
            <li><a href="/platform/embedding-indexes" className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">Similarity</a></li>
            <li><a href="/use-cases/ml-data-wrangling" className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">Export (ML)</a></li>
            <li><a href="/platform/data-sharing" className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">Share / Publish</a></li>
            <li><a href="/use-cases/agents-mcp" className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">Tools / MCP</a></li>
            <li><a href="/use-cases/ai-applications" className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">FastAPI</a></li>
          </ul>
        </div>
      </div>
    </div>
  )
}
