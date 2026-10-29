import { useEffect, useState } from 'react';
import { useDashboardRuntime } from './dashboard-runtime';

type MediaSource = { url: string; error?: string };

export function useMediaSource(source: string): MediaSource {
  const runtime = useDashboardRuntime();
  const resolveMedia = runtime.kind === 'embedded' ? runtime.resolveMedia : null;
  const [preview, setPreview] = useState<{ source: string; url: string; error?: string } | null>(null);
  useEffect(() => {
    if (!resolveMedia || !source) return;
    let active = true;
    resolveMedia(source).then(
      (url) => { if (active) setPreview({ source, url }); },
      (error: unknown) => { if (active) setPreview({ source, url: '', error: error instanceof Error ? error.message : 'The media preview could not be loaded.' }); },
    );
    return () => { active = false; };
  }, [source, resolveMedia]);
  if (!resolveMedia) return { url: source };
  return preview?.source === source ? preview : { url: '' };
}
