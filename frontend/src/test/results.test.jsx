// Offline check: every result view renders from real backend responses saved in fixtures.json.
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { METHODS } from "../methods.jsx";
import { ForestPlot } from "../ui.jsx";
import fixtures from "./fixtures.json";
import { CASES } from "./cases";

const view = (id) => METHODS.find((m) => m.id === id).Result;

describe("result views", () => {
  it.each(CASES.map(([id]) => id))("%s shows a plain-English conclusion", (id) => {
    const View = view(id);
    const { container } = render(<View r={fixtures[id]} />);
    expect(container.querySelector(".conclusion").textContent).toBe(
      (fixtures[id].summary ?? fixtures[id].size?.summary ?? fixtures[id].psm?.summary)
    );
  });

  it.each(["power", "bayes", "sequential", "peeking", "psm", "did"])("%s draws its chart", (id) => {
    const View = view(id);
    const { container } = render(<View r={fixtures[id]} />);
    expect(container.querySelector("svg.recharts-surface")).not.toBeNull();
  });

  it("psm draws the forest plot and the balance chart; did flags a failed placebo", () => {
    const Psm = view("psm"), Did = view("did");
    const psm = render(<Psm r={fixtures.psm} />).container;
    expect(psm.querySelectorAll("svg.forest circle").length).toBe(6);   // naive + matched + 4 sensitivity specs
    const did = render(<Did r={fixtures.did} />).container;
    expect(did.querySelector(".notes.bad")).not.toBeNull();
  });

  it("forest plot handles a single point without an interval and negative values", () => {
    const { container } = render(<ForestPlot rows={[{ label: "A", est: -635 }, { label: "B", est: 1131, lo: -689, hi: 3265 }]} />);
    expect(container.querySelectorAll("circle").length).toBe(2);
    expect(container.querySelectorAll(".ci").length).toBe(1);
  });
});
