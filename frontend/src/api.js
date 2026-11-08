import { errorMessage } from "./lib";

export const BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function request(path, { method = "POST", json, form } = {}) {
  let res;
  try {
    res = await fetch(BASE + path, {
      method,
      headers: json ? { "Content-Type": "application/json" } : undefined,
      body: json ? JSON.stringify(json) : form,
    });
  } catch {
    throw new Error(`Can't reach the backend at ${BASE}. Start it from the backend folder with: uvicorn app.main:app --reload`);
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error body */
  }
  if (!res.ok) throw new Error(errorMessage(body, res.status));
  return body;
}

export const api = {
  health: () => request("/health", { method: "GET" }),
  post: (path, json) => request(path, { json }),
  upload(file) {
    const form = new FormData();
    form.append("file", file);
    return request("/data/upload", { form });
  },
  lalonde: () => request("/data/sample/lalonde"),
  simulate: (json) => request("/data/simulate", { json }),
};
