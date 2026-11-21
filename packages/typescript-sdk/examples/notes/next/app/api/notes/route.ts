import createClient from 'openapi-fetch';
import type { paths } from '../../pixeltable';

export async function POST(request: Request) {
  const body: unknown = await request.json().catch(() => null);
  if (typeof body !== 'object' || body === null || !('title' in body) || typeof body.title !== 'string') {
    return Response.json({ message: 'Enter a title' }, { status: 400 });
  }

  const baseUrl = process.env.PIXELTABLE_SERVICE_URL;
  if (!baseUrl) return Response.json({ message: 'Set PIXELTABLE_SERVICE_URL' }, { status: 500 });

  const client = createClient<paths>({ baseUrl });
  const { data, response } = await client.POST('/notes', { body: { title: body.title } });
  if (!data) return Response.json({ message: 'Pixeltable could not save the note' }, { status: response.status });
  return Response.json(data);
}
