'use client';

import createClient from 'openapi-fetch';
import { useState, type FormEvent } from 'react';
import type { paths } from './pixeltable';

const client = createClient<paths>({ baseUrl: '/api' });
type Note = paths['/notes']['post']['responses'][200]['content']['application/json'];

export default function NotesPage() {
  const [note, setNote] = useState<Note>();
  const [error, setError] = useState('');

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const title = new FormData(event.currentTarget).get('title');
    if (typeof title !== 'string') return;
    setError('');
    setNote(undefined);
    try {
      const { data } = await client.POST('/notes', { body: { title } });
      if (!data) throw new Error('Could not save the note');
      setNote(data);
    } catch {
      setError('Could not save the note');
    }
  }

  return (
    <main>
      <h1>Notes</h1>
      <form onSubmit={submit}>
        <label>Title <input name="title" required /></label>
        <button type="submit">Save note</button>
      </form>
      <output>{note?.title_upper}</output>
      {error && <p role="alert">{error}</p>}
    </main>
  );
}
