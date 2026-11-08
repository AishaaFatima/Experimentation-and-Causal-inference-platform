// Offline check of the page flow with the backend mocked: nothing is shown until the choices are made.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import fixtures from "./fixtures.json";

vi.mock("../api", () => ({
  BASE: "http://backend.test",
  api: { health: vi.fn(), lalonde: vi.fn(), post: vi.fn(), simulate: vi.fn(), upload: vi.fn() },
}));
import { api } from "../api";
import App from "../App.jsx";

const cols = ["treat", "age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75", "re78"];
const dataset = {
  dataset_id: "d1", name: "lalonde.csv", n_rows: 614, notes: "Observational job-training data.", true_effect: null,
  columns: cols.map((name) => ({ name, dtype: "int64", n_unique: ["treat", "black", "hispan", "married", "nodegree"].includes(name) ? 2 : 50, n_missing: 0 })),
  preview: [Object.fromEntries(cols.map((c) => [c, 1]))],
};
const pickData = (v) => fireEvent.change(screen.getByLabelText(/Choose your data/), { target: { value: v } });
const pickMethod = (v) => fireEvent.change(screen.getByLabelText(/What do you want to do/), { target: { value: v } });

beforeEach(() => {
  vi.clearAllMocks();
  api.health.mockResolvedValue({ status: "ok" });
  api.lalonde.mockResolvedValue(dataset);
});

describe("page flow", () => {
  it("starts with a welcome message and no settings or run button", async () => {
    render(<App />);
    expect(await screen.findByText("Backend connected")).toBeTruthy();
    expect(screen.getByText(/Pick your data and what you want to do/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Run analysis" })).toBeNull();
    expect(screen.getByText("It's that tool")).toBeTruthy();
  });

  it("asks for data when the chosen analysis needs it, then lets you run once data is loaded", async () => {
    api.post.mockResolvedValue(fixtures.t_test);
    render(<App />);
    pickMethod("t_test");
    expect(screen.getByText(/needs data/i, { selector: ".blocked" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Run analysis" })).toBeNull();
    pickData("lalonde");
    await screen.findByText(/lalonde\.csv/);
    fireEvent.click(screen.getByRole("button", { name: "Run analysis" }));
    await waitFor(() => expect(document.querySelector(".conclusion")?.textContent).toBe(fixtures.t_test.summary));
    expect(api.post).toHaveBeenCalledWith("/ab/t-test", expect.objectContaining({ dataset_id: "d1", group_col: "treat", outcome_col: "re78" }));
  });

  it("an analysis that needs no data runs straight away, and switching methods clears the old result", async () => {
    api.post.mockImplementation(async (path) => (path === "/power/sample-size" ? fixtures.power.size : fixtures.power.curve));
    render(<App />);
    pickMethod("power");
    fireEvent.click(screen.getByRole("button", { name: "Run analysis" }));
    await waitFor(() => expect(document.querySelector(".conclusion")).not.toBeNull());
    pickMethod("peeking");
    expect(document.querySelector(".conclusion")).toBeNull();
  });

  it("pre-selects every characteristic for the causal methods", async () => {
    render(<App />);
    pickData("lalonde");
    await screen.findByText(/lalonde\.csv/);
    pickMethod("psm");
    const chosen = screen.getAllByRole("button").filter((b) => b.getAttribute("aria-pressed") === "true");
    expect(chosen.map((b) => b.textContent)).toEqual(["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]);
  });

  it("shows the backend's error message when a run fails", async () => {
    api.post.mockRejectedValue(new Error("Column 'zzz' not found"));
    render(<App />);
    pickData("lalonde");
    await screen.findByText(/lalonde\.csv/);
    pickMethod("t_test");
    fireEvent.click(screen.getByRole("button", { name: "Run analysis" }));
    expect(await screen.findByText("Column 'zzz' not found")).toBeTruthy();
  });
});
