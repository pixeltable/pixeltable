# Notes from TypeScript to Next.js

`app.ts` declares the Notes table and `POST /notes`. Pixeltable inserts a row and computes `title_upper`.
Next.js calls the service from its server route and renders the result. The prototype supports local JSON insert routes.

Use Node.js 22.18 or later and install `pixeltable[serve]` from this branch. From `packages/typescript-sdk`:

```bash
npm ci
demo=$(mktemp -d)
npm pack --pack-destination "$demo"
cp -R examples/notes "$demo/example"
mkdir "$demo/backend"
cp "$demo/example/app.ts" "$demo/backend/"
cd "$demo/backend"
printf '{"type":"module","private":true}\n' > package.json
printf '\n' > pixeltable.toml
printf 'node_modules/\n' > .gitignore
npm install "$demo/pixeltable-sdk-0.0.0-prototype.tgz"
pxt schema update app.ts notes_demo -f
pxt service update app.ts notes_demo -f
pxt service list notes_demo
```

Copy the service URL printed by `pxt service list`. Create the web app next to the backend:

```bash
cd "$demo"
npx create-next-app@latest web --ts --app --empty --use-npm --yes
cd web
npm install openapi-fetch
npm install -D openapi-typescript
cp -R "$demo/example/next/app/." app/
export PIXELTABLE_SERVICE_URL='http://127.0.0.1:<port>' # replace with your service URL
curl -fsS "$PIXELTABLE_SERVICE_URL/openapi.json" -o openapi.json
npx openapi-typescript openapi.json -o app/pixeltable.d.ts
npm run dev
```

Open the URL Next.js prints. Enter `hello from typescript` and click **Save note**. The page shows `HELLO FROM TYPESCRIPT`.

Add `title_length = this.title.len();` to the `Notes` class in `backend/app.ts`. Add `'title_length'` to the route's
`outputs`. From `backend`, run:

```bash
pxt schema update app.ts notes_demo -f
pxt service diff app.ts notes_demo
pxt service update app.ts notes_demo -f --allow-destructive
```

The service plan classifies changed outputs as a route replacement. The update restarts the demo service on the same
port. Download its OpenAPI schema and generate types again in `web`. Add `<span>{note?.title_length}</span>` next to
the uppercase title in `web/app/page.tsx`. Submit `hello` and check that the page shows `HELLO` and `5`.

Stop Next.js with Ctrl-C. From `backend`, run `pxt service stop notes_demo/notes`.
