// Runs every method against a REAL backend, exactly as the UI does, then renders its result view.
// Skipped automatically unless the backend is running on http://localhost:8000.
// To refresh the offline fixtures:  WRITE_FIXTURES=1 npm test   (PowerShell: $env:WRITE_FIXTURES=1; npm test)
import fs from "node:fs";
import path from "node:path";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "../App.jsx";
import { api } from "../api";
import { defaultsFor } from "../lib";
import { METHODS, initialValues } from "../methods.jsx";
import { CASES } from "./cases";

const live = await fetch("http://localhost:8000/health").then((r) => r.ok).catch(() => false);
const fixtures = {};

describe.skipIf(!live)("every method against the live backend", { timeout: 240000 }, () => {
  it.each(CASES)("%s", async (id, data) => {
    const method = METHODS.find((m) => m.id === id);
    const ds = data === "lalonde" ? await api.lalonde()
      : data === "sim" ? await api.simulate({ metric: "binary", n_per_group: 2000, lift: 0.03, baseline_rate: 0.1, seed: 1 })
      : null;
    const values = initialValues(method, defaultsFor(ds ? ds.columns : []), ds);
    const result = await method.run(values, ds);
    fixtures[id] = result;
    const { container } = render(<method.Result r={result} />);
    expect(container.querySelector(".conclusion")?.textContent.length).toBeGreaterThan(40);
    if (process.env.WRITE_FIXTURES) fs.writeFileSync(path.resolve(process.cwd(), "src/test/fixtures.json"), JSON.stringify(fixtures));
  });

  it("full flow in the app: pick the sample, compare averages, then match similar people", async () => {
    render(<App />);
    await screen.findByText("Backend connected");
    fireEvent.change(screen.getByLabelText(/Choose your data/), { target: { value: "lalonde" } });
    await screen.findByText(/lalonde\.csv/);
    fireEvent.change(screen.getByLabelText(/What do you want to do/), { target: { value: "t_test" } });
    fireEvent.click(screen.getByRole("button", { name: "Run analysis" }));
    await waitFor(() => expect(document.querySelector(".conclusion")?.textContent).toMatch(/-635/), { timeout: 20000 });
    fireEvent.change(screen.getByLabelText(/What do you want to do/), { target: { value: "psm" } });
    fireEvent.click(screen.getByRole("button", { name: "Run analysis" }));
    await waitFor(() => expect(document.querySelector(".conclusion")?.textContent).toMatch(/matching/i), { timeout: 30000 });
    expect(document.querySelectorAll("svg.forest circle").length).toBeGreaterThan(2);
  });

  it("shows a readable error when the analysis is invalid (rates test on a dollar outcome)", async () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText(/Choose your data/), { target: { value: "lalonde" } });
    await screen.findByText(/lalonde\.csv/);
    fireEvent.change(screen.getByLabelText(/What do you want to do/), { target: { value: "chi_square" } });
    fireEvent.click(screen.getByRole("button", { name: "Run analysis" }));
    const alert = await screen.findByText(/binary \(0\/1\)/, {}, { timeout: 10000 });
    expect(alert.closest("[role=alert]")).not.toBeNull();
  });
});
