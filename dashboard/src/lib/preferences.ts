export type DashboardPreferences = Pick<Storage, 'getItem' | 'setItem'>;

export const browserPreferences: DashboardPreferences = {
  getItem: (key) => localStorage.getItem(key),
  setItem: (key, value) => localStorage.setItem(key, value),
};

export function createMemoryPreferences(): DashboardPreferences {
  const values = new Map<string, string>();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value); },
  };
}
