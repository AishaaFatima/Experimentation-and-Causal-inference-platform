import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

afterEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("explains how to start the backend when it cannot be reached", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(api.post("/x", {})).rejects.toThrow(/uvicorn app\.main:app/);
  });
  it("turns a FastAPI validation error into a readable sentence", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: false, status: 422, json: async () => ({ detail: "Column 'zzz' not found" }) });
    await expect(api.post("/x", {})).rejects.toThrow("Column 'zzz' not found");
  });
  it("returns parsed JSON on success", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, status: 200, json: async () => ({ status: "ok" }) });
    await expect(api.health()).resolves.toEqual({ status: "ok" });
  });
});
