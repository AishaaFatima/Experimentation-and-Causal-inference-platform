# Frontend

React (Vite) interface for the experimentation and causal-inference backend.

## Run it

Two terminals, from the project root:

```powershell
# Terminal 1: backend
cd backend
.venv\\\\Scripts\\\\activate
uvicorn app.main:app --reload

# Terminal 2: frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The backend address defaults to http://localhost:8000; copy `.env.example` to `.env` to change it.

## Tests

```powershell
npm test
```

* Offline tests (formatting helpers, API client, every result view rendered from saved real responses) always run.
* Live tests call every method on the real backend, exactly as the UI does, and drive the app like a user.
They run automatically when the backend is up on port 8000 and are skipped otherwise.
* To refresh the saved responses: `$env:WRITE\\\_FIXTURES=1; npm test` (backend running).

## How it is organized

|File|Role|
|-|-|
|`src/methods.jsx`|One entry per analysis: its form, the backend call, and how its result is shown|
|`src/App.jsx`|The three steps (data, method, settings) and the result area|
|`src/ui.jsx`|Shared pieces, including the forest plot used by most methods|
|`src/lib.js`, `src/api.js`|Helpers and the backend client|



