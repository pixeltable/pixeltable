export const PipelineDiagram = () => {
  return (
    <div className="w-full bg-white dark:bg-neutral-900 p-3 rounded-lg border border-neutral-200 dark:border-neutral-700 not-prose overflow-x-auto">
      <div className="flex items-stretch gap-1.5 min-w-[680px]">
        {/* STORAGE Section */}
        <div className="w-[130px] flex-shrink-0 border border-neutral-300 dark:border-neutral-600 p-2.5 bg-white dark:bg-neutral-800 rounded">
          <h2 className="font-mono text-[11px] tracking-widest mb-1.5 uppercase text-neutral-900 dark:text-neutral-100 font-semibold">
            STORAGE
          </h2>
          <h3 className="font-mono text-[9px] mb-0.5 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
            Data Tables
          </h3>
          <div className="border-b border-neutral-200 dark:border-neutral-600 mb-1.5" />
          <ul className="space-y-0 text-[11px] font-mono text-neutral-900 dark:text-neutral-100">
            <li>• Structured</li>
            <li>• Video/Image</li>
            <li>• Audio/Doc</li>
            <li>• JSON/Text</li>
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
            ORCHESTRATION
          </h2>

          <div className="flex gap-3">
            {/* Pre-process */}
            <div className="flex-1">
              <h3 className="font-mono text-[9px] mb-0.5 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Pre-process
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-1" />
              <ul className="space-y-0 text-[11px] font-mono text-neutral-900 dark:text-neutral-100">
                <li>• Transform</li>
                <li>• Chunk/Split</li>
                <li>• Validate</li>
              </ul>
            </div>

            {/* Generate */}
            <div className="flex-1">
              <h3 className="font-mono text-[9px] mb-0.5 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Generate
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-1" />
              <ul className="space-y-0 text-[11px] font-mono text-neutral-900 dark:text-neutral-100 whitespace-nowrap">
                <li>• LLM calls</li>
                <li>• Local inference</li>
                <li>• Embeddings</li>
              </ul>
            </div>

            {/* Post-process */}
            <div className="flex-1">
              <h3 className="font-mono text-[9px] mb-0.5 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Post-process
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-1" />
              <ul className="space-y-0 text-[11px] font-mono text-neutral-900 dark:text-neutral-100">
                <li>• Aggregate</li>
                <li>• Extract</li>
                <li>• Index</li>
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
            RETRIEVAL
          </h2>
          <h3 className="font-mono text-[9px] mb-0.5 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
            Query & Serve
          </h3>
          <div className="border-b border-neutral-200 dark:border-neutral-600 mb-1.5" />
          <ul className="space-y-0 text-[11px] font-mono text-neutral-900 dark:text-neutral-100">
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
  )
}
