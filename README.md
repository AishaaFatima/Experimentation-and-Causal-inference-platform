# Platform details

Frontend React (Vite) interface for the experimentation and causal-inference backend.

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

## Screenshots of the project

Home Page
<img width="1480" height="638" alt="image" src="https://github.com/user-attachments/assets/27da7a80-d834-4732-9706-d730d43d0240" />

Choose your Dataset

<img width="757" height="322" alt="image" src="https://github.com/user-attachments/assets/eb93c9d2-8ff5-4cf9-a31d-b93351f6f1ff" />

Next, choose what to do with your dataset

<img width="1471" height="863" alt="image" src="https://github.com/user-attachments/assets/58f4710a-86b1-4c40-bfcb-2e961887a5f5" />

Run Analysis and acquire report
<img width="1428" height="1185" alt="image" src="https://github.com/user-attachments/assets/5335757c-f441-4d80-b826-a42e64ebd338" />

<img width="1436" height="767" alt="image" src="https://github.com/user-attachments/assets/23c1e4a7-09d0-44d9-9cb4-faff823125b3" />





