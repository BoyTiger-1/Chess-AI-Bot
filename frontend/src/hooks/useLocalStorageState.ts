import React from 'react';
import { readJson, writeJson } from '@/utils/storage';

export function useLocalStorageState<T>(key: string, initial: T) {
  const [value, setValue] = React.useState<T>(() => readJson<T>(key, initial));

  const setAndPersist = React.useCallback(
    (next: T | ((prev: T) => T)) => {
      setValue((prev) => {
        const resolved = typeof next === 'function' ? (next as (p: T) => T)(prev) : next;
        writeJson(key, resolved);
        return resolved;
      });
    },
    [key]
  );

  return [value, setAndPersist] as const;
}
