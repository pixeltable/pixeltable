import { createContext, useContext } from 'react';
import { createDashboardClient, type DashboardClient } from '@/api/client';
import { browserPreferences, type DashboardPreferences } from '@/lib/preferences';

type RuntimeBase = { client: DashboardClient; preferences: DashboardPreferences };
export type DashboardRuntime = RuntimeBase & (
  | { kind: 'local' }
  | { kind: 'embedded'; catalog: string; theme: 'light' | 'dark'; resolveMedia: (location: string) => Promise<string> }
);

export const DashboardRuntimeContext = createContext<DashboardRuntime>({
  kind: 'local', client: createDashboardClient(), preferences: browserPreferences,
});

export function useDashboardRuntime() {
  return useContext(DashboardRuntimeContext);
}
