import { describe, expect, it } from "vitest";
import { defaultsFor, errorMessage, fmtInt, fmtNum, fmtP, fmtPct, niceTicks, num, parseNumbers, wrapLabel } from "./lib";

const lalonde = ["treat", "age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75", "re78"].map((name) => ({
  name,
  dtype: "int64",
  n_unique: ["treat", "black", "hispan", "married", "nodegree"].includes(name) ? 2 : 50,
}));

describe("defaultsFor", () => {
  it("picks sensible columns for the LaLonde layout", () => {
    const d = defaultsFor(lalonde);
    expect(d.group).toBe("treat");
    expect(d.outcome).toBe("re78");
    expect(d.covariates).toEqual(["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]);
    expect([d.pre, d.post]).toEqual(["re75", "re78"]);
    expect([d.placeboPre, d.placeboPost]).toEqual(["re74", "re75"]);
  });
  it("handles a simulated A/B file and an empty dataset", () => {
    const d = defaultsFor([{ name: "group", dtype: "object", n_unique: 2 }, { name: "outcome", dtype: "float64", n_unique: 900 }]);
    expect([d.group, d.outcome, d.covariates]).toEqual(["group", "outcome", []]);
    expect(defaultsFor([]).group).toBe("");
  });
});

describe("parsing and formatting", () => {
  it("parses p-values", () => expect(parseNumbers("0.01, 0.02;0.5  0.9")).toEqual([0.01, 0.02, 0.5, 0.9]));
  it("rejects bad input", () => {
    expect(() => parseNumbers("0.1, abc")).toThrow(/numbers/);
    expect(() => parseNumbers("  ")).toThrow();
    expect(() => num("", "Baseline")).toThrow(/Baseline/);
    expect(num("0.5", "x")).toBe(0.5);
  });
  it("formats numbers readably", () => {
    expect(fmtNum(1131.4)).toBe("1,131");
    expect(fmtNum(-635.03)).toBe("-635");
    expect(fmtNum(12.345)).toBe("12.35");
    expect(fmtNum(0.0123)).toBe("0.012");
    expect(fmtNum(null)).toBe("–");
    expect(fmtInt(25)).toBe("25");
    expect(fmtInt(12345.4)).toBe("12,345");
    expect(fmtPct(0.99975)).toBe(">99.9%");
    expect(fmtPct(0.1762)).toBe("17.6%");
    expect(fmtP(0.0000001)).toBe("<0.001");
  });
  it("turns FastAPI errors into sentences", () => {
    expect(errorMessage({ detail: "Column 'x' not found" }, 422)).toBe("Column 'x' not found");
    expect(errorMessage({ detail: [{ loc: ["body", "mde"], msg: "Field required" }] }, 422)).toBe("mde: Field required");
    expect(errorMessage(null, 500)).toMatch(/500/);
  });
});

describe("niceTicks", () => {
  it("returns round ticks that include zero when the range spans it", () => {
    const t = niceTicks(-900, 3300);
    expect(t).toContain(0);
    expect(t[0]).toBeGreaterThanOrEqual(-900);
    expect(t[t.length - 1]).toBeLessThanOrEqual(3300);
  });
});

describe("wrapLabel", () => {
  it("keeps short labels on one line and wraps long ones to two", () => {
    expect(wrapLabel("Naive comparison")).toEqual(["Naive comparison"]);
    const two = wrapLabel("After matching (effect on the treated)");
    expect(two).toHaveLength(2);
    expect(two.every((l) => l.length <= 25)).toBe(true);
    expect(wrapLabel("Matching: + squared terms", 23)).toEqual(["Matching: + squared", "terms"]);
    expect(two.join(" ")).toBe("After matching (effect on the treated)");
  });
  it("shortens a very long second line", () => {
    const [, b] = wrapLabel("Doubly robust (on the treated) with an extremely long trailing description");
    expect(b.length).toBeLessThanOrEqual(25);
    expect(b.endsWith("…")).toBe(true);
  });
});
