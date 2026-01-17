export const PipelineDiagram = () => {
  return (
    <div className="w-full bg-white dark:bg-neutral-900 p-6 rounded-lg border border-neutral-200 dark:border-neutral-700 not-prose">
      <div className="flex flex-col lg:flex-row items-stretch gap-4">
        {/* STORAGE Section */}
        <div className="flex-shrink-0 border border-neutral-300 dark:border-neutral-600 p-5 bg-white dark:bg-neutral-800 rounded-lg lg:w-48">
          <h2 className="font-mono text-xs tracking-widest mb-4 uppercase text-neutral-900 dark:text-neutral-100 font-semibold">
            STORAGE
          </h2>

          <div className="mb-4">
            <h3 className="font-mono text-[10px] mb-2 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
              Data Tables
            </h3>
            <div className="border-b border-neutral-200 dark:border-neutral-600 mb-2" />
            <ul className="space-y-1.5 text-xs font-mono text-neutral-900 dark:text-neutral-100">
              <li>• Structured</li>
              <li>• Video/Image</li>
              <li>• Audio/Doc</li>
              <li>• JSON/Text</li>
            </ul>
          </div>

          <div className="space-y-0.5">
            <p className="font-mono text-[10px] text-neutral-500 dark:text-neutral-400 italic">
              Versioned
            </p>
            <p className="font-mono text-[10px] text-neutral-500 dark:text-neutral-400 italic">
              Cached
            </p>
          </div>
        </div>

        {/* Arrow 1 */}
        <div className="hidden lg:flex items-center flex-shrink-0 text-neutral-400 dark:text-neutral-500">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        </div>
        <div className="lg:hidden flex justify-center text-neutral-400 dark:text-neutral-500">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 5v14M5 13l7 7 7-7" />
          </svg>
        </div>

        {/* ORCHESTRATION Section */}
        <div className="flex-1 border border-neutral-300 dark:border-neutral-600 p-5 bg-white dark:bg-neutral-800 rounded-lg">
          <h2 className="font-mono text-xs tracking-widest mb-4 uppercase text-neutral-900 dark:text-neutral-100 font-semibold">
            ORCHESTRATION
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
            {/* Pre-process */}
            <div>
              <h3 className="font-mono text-[10px] mb-2 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Pre-process
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-2" />
              <ul className="space-y-1.5 text-xs font-mono text-neutral-900 dark:text-neutral-100">
                <li>• Transform</li>
                <li>• Chunk/Split</li>
                <li>• Validate</li>
              </ul>
            </div>

            {/* Generate */}
            <div>
              <h3 className="font-mono text-[10px] mb-2 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Generate
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-2" />
              <ul className="space-y-1.5 text-xs font-mono text-neutral-900 dark:text-neutral-100">
                <li>• LLM calls</li>
                <li>• Local inference</li>
                <li>• Embeddings</li>
              </ul>
            </div>

            {/* Post-process */}
            <div>
              <h3 className="font-mono text-[10px] mb-2 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
                Post-process
              </h3>
              <div className="border-b border-neutral-200 dark:border-neutral-600 mb-2" />
              <ul className="space-y-1.5 text-xs font-mono text-neutral-900 dark:text-neutral-100">
                <li>• Aggregate</li>
                <li>• Extract</li>
                <li>• Index</li>
              </ul>
            </div>
          </div>

          {/* Computed Columns Note */}
          <div className="border border-neutral-300 dark:border-neutral-600 p-3 bg-neutral-50 dark:bg-neutral-700 rounded">
            <p className="font-mono text-xs text-neutral-700 dark:text-neutral-200">
              Computed Columns: incremental, with lineage.
            </p>
          </div>
        </div>

        {/* Arrow 2 */}
        <div className="hidden lg:flex items-center flex-shrink-0 text-neutral-400 dark:text-neutral-500">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M5 12h14M13 5l7 7-7 7" />
          </svg>
        </div>
        <div className="lg:hidden flex justify-center text-neutral-400 dark:text-neutral-500">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 5v14M5 13l7 7 7-7" />
          </svg>
        </div>

        {/* RETRIEVAL Section */}
        <div className="flex-shrink-0 border border-neutral-300 dark:border-neutral-600 p-5 bg-white dark:bg-neutral-800 rounded-lg lg:w-48">
          <h2 className="font-mono text-xs tracking-widest mb-4 uppercase text-neutral-900 dark:text-neutral-100 font-semibold">
            RETRIEVAL
          </h2>

          <div>
            <h3 className="font-mono text-[10px] mb-2 uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
              Query & Serve
            </h3>
            <div className="border-b border-neutral-200 dark:border-neutral-600 mb-2" />
            <ul className="space-y-1.5 text-xs font-mono text-neutral-900 dark:text-neutral-100">
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
