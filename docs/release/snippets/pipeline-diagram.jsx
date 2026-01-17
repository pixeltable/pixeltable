export const PipelineDiagram = () => {
  return (
    <div className="w-full bg-white dark:bg-neutral-900 p-4 rounded-lg border border-neutral-200 dark:border-neutral-700 not-prose overflow-x-auto">
      <div className="flex items-stretch gap-3 min-w-[800px]">
        {/* STORAGE Section */}
        <div className="w-[180px] flex-shrink-0 border border-neutral-300 dark:border-neutral-600 p-4 bg-white dark:bg-neutral-800 rounded">
          <h2 className="font-mono text-[11px] tracking-widest mb-3 uppercase text-neutral-900 dark:text-neutral-100 font-semibold">
            STORAGE
          </h2>

          <div className="mb-3">
            <h3 className="font-mono text-[10px] mb-1 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
              Data Tables
            </h3>
            <div className="border-b border-neutral-200 dark:border-neutral-600 mb-2" />
            <ul className="space-y-1 text-[11px] font-mono text-neutral-900 dark:text-neutral-100">
              <li>• Structured</li>
              <li>• Video/Image</li>
              <li>• Audio/Doc</li>
              <li>• JSON/Text</li>
            </ul>
          </div>

          <div className="space-y-0.5 mt-4">
            <p className="font-mono text-[10px] text-neutral-500 dark:text-neutral-400 italic">Versioned</p>
            <p className="font-mono text-[10px] text-neutral-500 dark:text-neutral-400 italic">Cached</p>
          </div>
        </div>

        {/* Arrow 1 */}
        <div className="flex items-center flex-shrink-0 text-neutral-400 dark:text-neutral-500">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        </div>

        {/* ORCHESTRATION Section */}
        <div className="flex-1 min-w-[380px] border border-neutral-300 dark:border-neutral-600 p-4 bg-white dark:bg-neutral-800 rounded">
          <h2 className="font-mono text-[11px] tracking-widest mb-3 uppercase text-neutral-900 dark:text-neutral-100 font-semibold">
            ORCHESTRATION
          </h2>

          <div className="flex gap-6 mb-3">
            {/* Pre-process */}
            <div className="flex-1">
              <h3 className="font-mono text-[10px] mb-1 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Pre-process
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-2" />
              <ul className="space-y-1 text-[11px] font-mono text-neutral-900 dark:text-neutral-100">
                <li>• Transform</li>
                <li>• Chunk/Split</li>
                <li>• Validate</li>
              </ul>
            </div>

            {/* Generate */}
            <div className="flex-1">
              <h3 className="font-mono text-[10px] mb-1 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Generate
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-2" />
              <ul className="space-y-1 text-[11px] font-mono text-neutral-900 dark:text-neutral-100 whitespace-nowrap">
                <li>• LLM calls</li>
                <li>• Local inference</li>
                <li>• Embeddings</li>
              </ul>
            </div>

            {/* Post-process */}
            <div className="flex-1">
              <h3 className="font-mono text-[10px] mb-1 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Post-process
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-2" />
              <ul className="space-y-1 text-[11px] font-mono text-neutral-900 dark:text-neutral-100">
                <li>• Aggregate</li>
                <li>• Extract</li>
                <li>• Index</li>
              </ul>
            </div>
          </div>

          {/* Computed Columns Note */}
          <div className="border border-neutral-300 dark:border-neutral-600 px-3 py-2 bg-neutral-50 dark:bg-neutral-700 rounded">
            <p className="font-mono text-[11px] text-neutral-700 dark:text-neutral-200">
              Computed Columns: incremental, with lineage.
            </p>
          </div>
        </div>

        {/* Arrow 2 */}
        <div className="flex items-center flex-shrink-0 text-neutral-400 dark:text-neutral-500">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        </div>

        {/* RETRIEVAL Section */}
        <div className="w-[180px] flex-shrink-0 border border-neutral-300 dark:border-neutral-600 p-4 bg-white dark:bg-neutral-800 rounded">
          <h2 className="font-mono text-[11px] tracking-widest mb-3 uppercase text-neutral-900 dark:text-neutral-100 font-semibold">
            RETRIEVAL
          </h2>

          <div>
            <h3 className="font-mono text-[10px] mb-1 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
              Query & Serve
            </h3>
            <div className="border-b border-neutral-200 dark:border-neutral-600 mb-2" />
            <ul className="space-y-1 text-[11px] font-mono text-neutral-900 dark:text-neutral-100">
              <li>• SQL-like</li>
              <li>• Similarity</li>
              <li>• Export (ML)</li>
              <li>• Share/Publish</li>
              <li>• Tools/MCP</li>
              <li>• FastAPI</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
