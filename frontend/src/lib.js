// Small pure helpers (unit-tested in lib.test.js)

export function errorMessage(body, status) {
  const d = body && body.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d.map((e) => `${(e.loc || []).filter((x) => x !== "body").join(".")}: ${e.msg}`).join("; ");
  }
  return `The request failed (HTTP ${status}).`;
}

export function parseNumbers(text) {
  const nums = text.split(/[\s,;]+/).filter(Boolean).map(Number);
  if (!nums.length || nums.some(Number.isNaN)) {
    throw new Error("Enter numbers separated by commas or spaces, for example 0.001, 0.02, 0.04.");
  }
  return nums;
}

export function num(value, label) {
  const n = Number(value);
  if (value === "" || value === null || value === undefined || Number.isNaN(n)) {
    throw new Error(`Enter a number for "${label}".`);
  }
  return n;
}

export function fmtNum(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  const a = Math.abs(x);
  if (a >= 1000) return Math.round(x).toLocaleString("en-US");
  if (a >= 100) return x.toFixed(0);
  if (a >= 1) return x.toFixed(2);
  if (a === 0) return "0";
  return Number(x.toPrecision(2)).toString();
}

export const fmtInt = (n) => Math.round(n).toLocaleString("en-US");

export function fmtPct(p) {
  if (p === null || p === undefined) return "–";
  if (p > 0.999) return ">99.9%";
  if (p < 0.001) return "<0.1%";
  return `${(p * 100).toFixed(1)}%`;
}

export function fmtP(p) {
  if (p === null || p === undefined) return "–";
  return p < 0.001 ? "<0.001" : p.toFixed(3);
}

export function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) {
    ticks.push(Math.abs(t) < step * 1e-9 ? 0 : Number(t.toPrecision(12)));
  }
  return ticks;
}

const isNumeric = (c) => /int|float/.test(c.dtype);
export { isNumeric };

/** Guess sensible column choices for a freshly loaded dataset. */
export function defaultsFor(columns = []) {
  const numeric = columns.filter(isNumeric);
  const twoValued = columns.filter((c) => c.n_unique === 2);
  const group =
    (twoValued.find((c) => /treat|group|variant|arm|assign|test/i.test(c.name)) || twoValued[0] || columns[0] || {}).name || "";
  const candidates = numeric.filter((c) => c.name !== group);
  const outcome =
    (candidates.find((c) => /outcome|convert|revenue|^y$|re78|metric/i.test(c.name)) || candidates[candidates.length - 1] || {}).name || "";
  const order = candidates.map((c) => c.name);
  const oi = order.indexOf(outcome);
  return {
    group,
    outcome,
    covariates: order.filter((n) => n !== outcome),
    pre: oi > 0 ? order[oi - 1] : "",
    post: outcome,
    placeboPre: oi > 1 ? order[oi - 2] : "",
    placeboPost: oi > 0 ? order[oi - 1] : "",
  };
}

/** Split a long chart label into at most two lines of roughly `max` characters. */
export function wrapLabel(label, max = 25) {
  if (label.length <= max) return [label];
  const cut = label.lastIndexOf(" ", max);
  const i = cut > 8 ? cut : max;
  const first = label.slice(0, i).trim();
  let second = label.slice(i).trim();
  if (second.length > max) second = `${second.slice(0, max - 1)}…`;
  return [first, second];
}
