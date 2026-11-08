import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "./api";
import { fmtInt, fmtNum, fmtP, fmtPct, niceTicks, num, parseNumbers } from "./lib";
import { Badge, ChartBox, Conclusion, ForestPlot, Notes, Stats, Table } from "./ui";

const ACCENT = "#3347c9", BAD = "#b03d27", INK2 = "#455566", LINE = "#d5dbe3";
const ALPHAS = [["0.05", "5%"], ["0.01", "1%"], ["0.1", "10%"]];
const pts = (v) => `${(v * 100).toFixed(1)}`;           // proportion -> percentage points
const tipStyle = { borderRadius: 8, border: `1px solid ${LINE}`, fontSize: 13 };

const groupCol = { key: "group_col", label: "Group column", help: "Who is in which group.", type: "column", kind: "any", default: (d) => d.group };
const outcomeCol = { key: "outcome_col", label: "Outcome column", type: "column", kind: "numeric", default: (d) => d.outcome };
const treatCol = { key: "treat_col", label: "Treatment column", help: "Who was treated (0 or 1).", type: "column", kind: "any", default: (d) => d.group };
const covariates = { key: "covariates", label: "Characteristics to account for", type: "columns",
  help: "Pick everything that could influence both who gets treated and the outcome.", default: (d) => d.covariates };
const alpha = { key: "alpha", label: "Significance level", type: "select", options: ALPHAS, default: "0.05" };
const base = (v, ds) => ({ dataset_id: ds.dataset_id, group_col: v.group_col, outcome_col: v.outcome_col, alpha: num(v.alpha, "Significance level") });
const causalBase = (v, ds) => ({ dataset_id: ds.dataset_id, treat_col: v.treat_col, outcome_col: v.outcome_col, covariates: v.covariates });
const needCovs = (v) => { if (!v.covariates.length) throw new Error("Pick at least one characteristic to account for."); };

/* ---------- result views ---------- */

function PowerResult({ r }) {
  const { size, curve, input } = r;
  const ns = curve.curve.map((c) => c.n_per_group);
  const ticks = niceTicks(Math.min(...ns), Math.max(...ns)).map(Math.round);
  return (
    <>
      <Conclusion>{size.summary}</Conclusion>
      <Stats items={[["Per group", size.n_control.toLocaleString()], ["In total", size.n_total.toLocaleString()], ["Standardized effect", size.effect_size.toFixed(3)]]} />
      <ChartBox title="How power grows with sample size" caption="Power is the chance of detecting the effect if it is real. The dashed lines mark your target power and the sample size that reaches it.">
        <ResponsiveContainer width="100%" height={270}>
          <LineChart data={curve.curve} margin={{ top: 10, right: 20, bottom: 18, left: 0 }}>
            <CartesianGrid stroke={LINE} vertical={false} />
            <XAxis type="number" dataKey="n_per_group" domain={["dataMin", "dataMax"]} ticks={ticks} tickFormatter={fmtInt} label={{ value: "People per group", position: "insideBottom", offset: -10 }} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} labelFormatter={(n) => `${fmtInt(n)} per group`} contentStyle={tipStyle} />
            <ReferenceLine y={input.power} stroke={INK2} strokeDasharray="4 3" />
            <ReferenceLine x={curve.required_n_per_group} stroke={INK2} strokeDasharray="4 3" />
            <Line isAnimationActive={false} type="monotone" dataKey="power" stroke={ACCENT} strokeWidth={2.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartBox>
    </>
  );
}

function MultipleResult({ r }) {
  return (
    <>
      <Conclusion>{r.summary}</Conclusion>
      <Table
        cols={[
          { key: "label", label: "Test" },
          { key: "p_value", label: "Raw p-value", num: true, render: (x) => fmtP(x.p_value) },
          { key: "adjusted_p", label: "Adjusted p-value", num: true, render: (x) => fmtP(x.adjusted_p) },
          { key: "reject_null", label: "After correction", render: (x) => <Badge tone={x.reject_null ? "good" : "neutral"}>{x.reject_null ? "Significant" : "Not significant"}</Badge> },
        ]}
        rows={r.results}
      />
    </>
  );
}

function TwoGroupResult({ r, binary }) {
  const fmt = binary ? pts : fmtNum;
  return (
    <>
      <Conclusion>{r.summary}</Conclusion>
      <Stats items={binary
        ? [["Control rate", fmtPct(r.rate_control)], ["Treatment rate", fmtPct(r.rate_treatment)], ["p-value", fmtP(r.p_value)], ["Test used", r.method === "fisher_exact" ? "Fisher's exact" : "Chi-square"]]
        : [["Control average", fmtNum(r.mean_control)], ["Treatment average", fmtNum(r.mean_treatment)], ["p-value", fmtP(r.p_value)], ["Effect size (d)", r.cohens_d === null ? "–" : r.cohens_d.toFixed(2)]]} />
      <Notes items={r.warnings} />
      <ChartBox title="Difference between groups" caption={`Treatment minus control${binary ? ", in percentage points" : ""}. If the interval crosses the dashed line at zero, the data cannot rule out no difference. This comparison is only trustworthy if people were assigned to groups at random.`}>
        <ForestPlot format={fmt} rows={[{ label: "Treatment − control", est: r.difference, lo: r.ci_lower, hi: r.ci_upper }]} />
      </ChartBox>
    </>
  );
}

function BayesResult({ r }) {
  const binary = r.metric === "binary";
  const fmt = binary ? pts : fmtNum;
  const data = r.histograms.difference.map((b) => ({ x: (b.x_start + b.x_end) / 2, density: b.density }));
  const d = r.posterior.difference;
  const xs = data.map((p) => p.x);
  const ticks = niceTicks(Math.min(...xs), Math.max(...xs));
  return (
    <>
      <Conclusion>{r.summary}</Conclusion>
      <Stats items={[["Chance treatment is better", fmtPct(r.prob_treatment_better)], ["Most likely difference", binary ? `${pts(d.mean)} pts` : fmtNum(d.mean)], ["Loss if you pick treatment", binary ? `${(r.expected_loss_choose_treatment * 100).toFixed(2)} pts` : fmtNum(r.expected_loss_choose_treatment)]]} />
      <Notes items={r.warnings} />
      <Notes tone="info" items={[r.caveat]} />
      <ChartBox title="What the data say the true difference could be" caption="Taller means more plausible. The share of the area to the right of zero is the chance the treatment is better.">
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={data} margin={{ top: 10, right: 20, bottom: 18, left: 0 }}>
            <CartesianGrid stroke={LINE} vertical={false} />
            <XAxis type="number" dataKey="x" domain={["dataMin", "dataMax"]} ticks={ticks} tickFormatter={fmt} label={{ value: binary ? "Difference in percentage points" : "Difference in the outcome", position: "insideBottom", offset: -10 }} />
            <YAxis hide />
            <Tooltip formatter={(v) => v.toFixed(2)} labelFormatter={(x) => `Difference ≈ ${fmt(x)}`} contentStyle={tipStyle} />
            <ReferenceLine x={0} stroke="#16212e" strokeDasharray="4 3" />
            <Area isAnimationActive={false} type="monotone" dataKey="density" stroke={ACCENT} fill={ACCENT} fillOpacity={0.22} strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartBox>
    </>
  );
}

function SequentialResult({ r }) {
  const data = r.bounds.z_bounds.map((b, i) => ({ look: i + 1, boundary: b, z: r.looks[i].status === "analyzed" ? Math.abs(r.looks[i].z) : null }));
  return (
    <>
      <Conclusion>{r.summary}</Conclusion>
      <ChartBox title="Evidence at each planned check" caption="Stop only when the evidence line rises above the boundary. Early boundaries are strict on purpose, which is what keeps repeated checking from producing false winners.">
        <ResponsiveContainer width="100%" height={270}>
          <LineChart data={data} margin={{ top: 10, right: 20, bottom: 18, left: 0 }}>
            <CartesianGrid stroke={LINE} vertical={false} />
            <XAxis dataKey="look" label={{ value: "Planned check", position: "insideBottom", offset: -10 }} />
            <YAxis tickFormatter={(v) => v.toFixed(1)} />
            <Tooltip formatter={(v) => (v === null ? "not reached" : v.toFixed(2))} contentStyle={tipStyle} />
            <Legend verticalAlign="top" height={40} />
            <Line isAnimationActive={false} type="stepAfter" dataKey="boundary" name="Stopping boundary" stroke={BAD} strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} />
            <Line isAnimationActive={false} type="monotone" dataKey="z" name="Evidence so far (|z|)" stroke={ACCENT} strokeWidth={2.5} dot={{ r: 4 }} connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartBox>
    </>
  );
}

function PeekingResult({ r }) {
  const data = [
    { name: "Check every time, stop at the first p < 5%", rate: r.naive_peeking, bad: true },
    { name: "O'Brien-Fleming boundaries", rate: r.obrien_fleming },
    { name: "Pocock boundaries", rate: r.pocock },
  ];
  return (
    <>
      <Conclusion>{r.summary}</Conclusion>
      <ChartBox title="How often a test with no real effect declares a winner" caption={`Based on ${r.n_sims.toLocaleString()} simulated experiments in which nothing changed. The dashed line is the ${(r.alpha * 100).toFixed(0)}% error rate you thought you were getting.`}>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={data} layout="vertical" margin={{ top: 8, right: 30, bottom: 8, left: 10 }}>
            <CartesianGrid stroke={LINE} horizontal={false} />
            <XAxis type="number" domain={[0, "dataMax"]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <YAxis type="category" dataKey="name" width={250} />
            <Tooltip cursor={{ fill: "#edf0f4" }} formatter={(v) => `${(v * 100).toFixed(1)}%`} contentStyle={tipStyle} />
            <ReferenceLine x={r.alpha} stroke="#16212e" strokeDasharray="4 3" />
            <Bar isAnimationActive={false} dataKey="rate" name="False-positive rate" radius={[0, 4, 4, 0]}>
              {data.map((d) => <Cell key={d.name} fill={d.bad ? BAD : ACCENT} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartBox>
    </>
  );
}

const SPEC_LABEL = { "linear terms only": "Matching: simple model", "+ zero-earnings flags": "Matching: + zero flags", "+ squared terms": "Matching: + squared terms", "+ squares and zero flags": "Matching: + squares and flags" };

function PsmResult({ r: { psm: r, sens } }) {
  const bal = [...r.balance].sort((a, b) => Math.abs(b.smd_before) - Math.abs(a.smd_before)).map((b) => ({ name: b.covariate, before: Math.abs(b.smd_before), after: Math.abs(b.smd_after) }));
  const maxBefore = Math.max(...bal.map((b) => b.before));
  const balTop = Math.ceil(maxBefore * 5) / 5;
  const rows = [
    { label: "Naive comparison", est: r.naive_difference, tone: "naive" },
    { label: "After matching (effect on the treated)", est: r.att, lo: r.ci_lower, hi: r.ci_upper },
    ...(sens ? sens.specifications.map((s) => ({ label: (SPEC_LABEL[s.specification] || s.specification) + (s.balanced ? "" : " (imbalanced)"), est: s.att, tone: "muted" })) : []),
  ];
  return (
    <>
      <Conclusion>{r.summary}</Conclusion>
      <Stats items={[["Treated people matched", `${r.n_treated_matched} of ${r.n_treated}`], ["Comparison people used", r.n_controls_used], ["Worst imbalance before", r.max_abs_smd_before.toFixed(2)], ["Worst imbalance after", r.max_abs_smd_after.toFixed(2)]]} />
      <Notes items={sens ? [sens.summary] : []} />
      <ChartBox title="The estimate, and how much it depends on modelling choices" caption="Red is the naive comparison, which ignores differences between the groups. Blue is the matched estimate with its confidence interval. The grey dots re-run the matching with other reasonable model choices; those marked imbalanced did not make the groups comparable, so trust them less.">
        <ForestPlot rows={rows} />
      </ChartBox>
      <ChartBox title="Did matching make the groups comparable?" caption="Each bar is how far apart the two groups are on that characteristic, in standard deviations. Below the dashed line at 0.1 counts as balanced.">
        <ResponsiveContainer width="100%" height={60 + bal.length * 34}>
          <BarChart data={bal} layout="vertical" margin={{ top: 8, right: 30, bottom: 8, left: 10 }}>
            <CartesianGrid stroke={LINE} horizontal={false} />
            <XAxis type="number" domain={[0, balTop]} ticks={niceTicks(0, balTop, 4)} tickFormatter={(v) => v.toFixed(1)} />
            <YAxis type="category" dataKey="name" width={80} />
            <Tooltip cursor={{ fill: "#edf0f4" }} formatter={(v) => v.toFixed(2)} contentStyle={tipStyle} />
            <Legend verticalAlign="top" height={40} />
            <ReferenceLine x={0.1} stroke="#16212e" strokeDasharray="4 3" />
            <Bar isAnimationActive={false} dataKey="before" name="Before matching" fill="#9aa6b5" />
            <Bar isAnimationActive={false} dataKey="after" name="After matching" fill={ACCENT} />
          </BarChart>
        </ResponsiveContainer>
      </ChartBox>
    </>
  );
}

function DidResult({ r }) {
  const pl = r.placebo;
  const data = [
    { period: r.pre_label, Treated: r.means.treated_pre, Comparison: r.means.control_pre },
    { period: r.post_label, Treated: r.means.treated_post, Comparison: r.means.control_post },
  ];
  const rows = [{ label: "Difference-in-differences", est: r.estimate, lo: r.ci_lower, hi: r.ci_upper }];
  if (pl) rows.push({ label: `Placebo check (${pl.pre_col} → ${pl.post_col})`, est: pl.estimate, lo: pl.ci_lower, hi: pl.ci_upper, tone: pl.parallel_trends_plausible ? "muted" : "naive" });
  return (
    <>
      <Conclusion>{r.summary}</Conclusion>
      {pl && <Notes tone={pl.parallel_trends_plausible ? "info" : "bad"} items={[pl.parallel_trends_plausible ? "The placebo check passed: the groups were not already drifting apart before the treatment." : "The placebo check failed: the groups were already on different paths before the treatment, so this method's key assumption looks violated."]} />}
      <ChartBox title="Average outcome before and after" caption="This method assumes the two lines would have moved in parallel without the treatment. The gap between how much they actually moved is the estimate.">
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={data} margin={{ top: 10, right: 30, bottom: 8, left: 10 }}>
            <CartesianGrid stroke={LINE} vertical={false} />
            <XAxis dataKey="period" padding={{ left: 40, right: 40 }} />
            <YAxis tickFormatter={fmtNum} />
            <Tooltip formatter={(v) => fmtNum(v)} contentStyle={tipStyle} />
            <Legend verticalAlign="top" height={40} />
            <Line isAnimationActive={false} type="linear" dataKey="Treated" stroke={ACCENT} strokeWidth={2.5} dot={{ r: 5 }} />
            <Line isAnimationActive={false} type="linear" dataKey="Comparison" stroke="#7a8796" strokeWidth={2.5} dot={{ r: 5 }} />
          </LineChart>
        </ResponsiveContainer>
      </ChartBox>
      <ChartBox title="The estimate, with a check for pre-existing trends" caption="The placebo row applies the same method to two earlier periods, before any treatment. It should land near zero. If it does not, the method is picking up differences that were already there.">
        <ForestPlot rows={rows} />
      </ChartBox>
    </>
  );
}

const DOWHY_LABEL = { linear_regression: "Regression adjustment", propensity_score_weighting: "Propensity weighting", propensity_score_matching: "DoWhy matching" };

function DowhyResult({ r }) {
  const rows = [{ label: "Naive comparison", est: r.naive_difference, tone: "naive" },
    ...r.estimates.filter((e) => e.att !== null).map((e) => ({ label: DOWHY_LABEL[e.method] || e.method, est: e.att }))];
  return (
    <>
      <Conclusion>{r.summary}</Conclusion>
      <Notes tone="info" items={[`Assumption: ${r.identification.assumption}`]} />
      <ChartBox title="Three ways of adjusting for the same characteristics" caption="If different methods land in a similar range, that is reassuring. If they disagree, report the range rather than one number.">
        <ForestPlot rows={rows} />
      </ChartBox>
      <ChartBox title={`Attempts to break the ${(DOWHY_LABEL[r.refuted_estimator] || "").toLowerCase()} estimate`} caption="A result that survives these checks is not fragile. That is weaker than proof: none of them can detect a confounder you never measured.">
        <Table
          cols={[
            { key: "title", label: "Check" },
            { key: "orig", label: "Estimate before", num: true, render: (x) => fmtNum(x.original_effect) },
            { key: "new", label: "Estimate after", num: true, render: (x) => fmtNum(x.new_effect) },
            { key: "res", label: "Result", render: (x) => <Badge tone={x.passed === null ? "neutral" : x.passed ? "good" : "bad"}>{x.passed === null ? "Not run" : x.passed ? "Held up" : "Failed"}</Badge> },
          ]}
          rows={r.refutations}
        />
      </ChartBox>
    </>
  );
}

const ECON_LABEL = { linear_dml: "Double ML (one average effect)", causal_forest: "Causal forest (on the treated)", doubly_robust: "Doubly robust (on the treated)" };

function EconmlResult({ r }) {
  const rows = [{ label: "Naive comparison", est: r.naive_difference, tone: "naive" },
    ...r.estimates.map((e) => ({ label: ECON_LABEL[e.method] || e.method, est: e.effect, lo: e.ci_lower ?? undefined, hi: e.ci_upper ?? undefined }))];
  const h = r.heterogeneity;
  return (
    <>
      <Conclusion>{r.summary}</Conclusion>
      <Notes items={r.warnings} />
      <ChartBox title="Machine-learning estimates of the effect" caption="The doubly robust estimate has no interval. See below for how far it moves when extreme probabilities are clipped differently. The causal forest interval is wide because it is deliberately conservative.">
        <ForestPlot rows={rows} />
      </ChartBox>
      <ChartBox title="Doubly robust estimate at different clipping levels" caption="Clipping limits how much weight any one person can carry. A large spread means this estimate should not be trusted on its own.">
        <Table cols={[{ key: "min_propensity", label: "Clipped to at least" }, { key: "att", label: "Effect on the treated", num: true, render: (x) => fmtNum(x.att) }]} rows={r.dr_sensitivity} />
      </ChartBox>
      <ChartBox title="Who seems to benefit most?" caption={h.caution}>
        <Table cols={[{ key: "group", label: "Group (treated people only)" }, { key: "n", label: "People", num: true, render: (x) => x.n_treated }, { key: "e", label: "Average effect", num: true, render: (x) => fmtNum(x.mean_effect) }]} rows={h.subgroups} />
      </ChartBox>
    </>
  );
}

/* ---------- registry ---------- */

export const GROUPS = [
  { id: "plan", title: "Plan a test" },
  { id: "ab", title: "Analyze an A/B test" },
  { id: "causal", title: "Estimate cause and effect without an experiment" },
];

export const METHODS = [
  {
    id: "power", group: "plan", label: "How many people do I need?", blurb: "Sample size and power for a planned test.", needsData: false,
    fields: [
      { key: "metric", label: "What are you measuring?", type: "select", options: [["binary", "A rate (yes or no)"], ["continuous", "An average (revenue, time)"]], default: "binary" },
      { key: "baseline_rate", label: "Current rate", type: "number", step: "0.01", help: "For example 0.10 for 10%.", default: "0.10", showIf: (v) => v.metric === "binary" },
      { key: "std", label: "Spread (standard deviation)", type: "number", step: "any", default: "20", showIf: (v) => v.metric === "continuous" },
      { key: "mde", label: "Smallest change to detect", type: "number", step: "any", help: (v) => (v.metric === "binary" ? "A change in the rate: 0.02 means +2 percentage points." : "In the same units as your outcome: 5 means a difference of 5."), default: "0.02" },
      alpha,
      { key: "power", label: "Power", type: "select", options: [["0.8", "80%"], ["0.9", "90%"], ["0.95", "95%"]], default: "0.8" },
    ],
    onChange: (k, val) => (k === "metric" ? { mde: val === "binary" ? "0.02" : "5" } : {}),
    async run(v) {
      const body = { metric: v.metric, mde: num(v.mde, "Smallest change"), alpha: num(v.alpha, "Significance level"), power: num(v.power, "Power") };
      if (v.metric === "binary") body.baseline_rate = num(v.baseline_rate, "Current rate"); else body.std = num(v.std, "Spread");
      const [size, curve] = await Promise.all([api.post("/power/sample-size", body), api.post("/power/curve", body)]);
      return { size, curve, input: body };
    },
    Result: PowerResult,
  },
  {
    id: "multiple", group: "ab", label: "Many tests at once", blurb: "Correct p-values when you test several things.", needsData: false,
    fields: [
      { key: "p_values", label: "Your p-values", type: "text", wide: true, help: "Separated by commas or spaces.", default: "0.001, 0.012, 0.03, 0.04, 0.2, 0.6" },
      { key: "method", label: "Correction", type: "select", options: [["benjamini_hochberg", "Benjamini-Hochberg (controls false discoveries)"], ["bonferroni", "Bonferroni (strictest)"]], default: "benjamini_hochberg" },
      alpha,
    ],
    run: (v) => api.post("/ab/multiple-testing", { p_values: parseNumbers(v.p_values), method: v.method, alpha: num(v.alpha, "Significance level") }),
    Result: MultipleResult,
  },
  {
    id: "t_test", group: "ab", label: "Compare averages", blurb: "t-test for revenue, time, scores.", needsData: true,
    fields: [groupCol, outcomeCol, alpha],
    run: (v, ds) => api.post("/ab/t-test", base(v, ds)),
    Result: (p) => <TwoGroupResult {...p} binary={false} />,
  },
  {
    id: "chi_square", group: "ab", label: "Compare rates", blurb: "Chi-square test for conversions (0/1 outcome).", needsData: true,
    fields: [groupCol, outcomeCol, alpha],
    run: (v, ds) => api.post("/ab/chi-square", base(v, ds)),
    Result: (p) => <TwoGroupResult {...p} binary />,
  },
  {
    id: "bayes", group: "ab", label: "Chance B beats A", blurb: "Bayesian A/B test, in plain probabilities.", needsData: true, slow: true,
    fields: [groupCol, outcomeCol, { key: "draws", label: "Accuracy", type: "select", options: [["2000", "Quick (2,000 samples)"], ["4000", "Standard (4,000 samples)"], ["10000", "Precise (10,000 samples)"]], default: "4000" }],
    run: (v, ds) => api.post("/bayes/ab", { dataset_id: ds.dataset_id, group_col: v.group_col, outcome_col: v.outcome_col, draws: num(v.draws, "Accuracy") }),
    Result: BayesResult,
  },
  {
    id: "sequential", group: "ab", label: "Check results safely mid-test", blurb: "Sequential testing so peeking is safe.", needsData: true,
    fields: [
      groupCol, outcomeCol,
      { key: "planned_total_n", label: "People you plan to enrol", type: "number", step: "1", help: "Rows in the file are treated as arriving in order.", default: (d, ds) => String(ds ? ds.n_rows : 1000) },
      { key: "n_looks", label: "Planned checks", type: "select", options: [["3", "3"], ["5", "5"], ["8", "8"], ["10", "10"]], default: "5" },
      { key: "spending", label: "Boundary style", type: "select", options: [["obrien_fleming", "O'Brien-Fleming (strict early, lenient late)"], ["pocock", "Pocock (equal at every check)"]], default: "obrien_fleming" },
      alpha,
    ],
    run: (v, ds) => api.post("/sequential/analyze", { ...base(v, ds), planned_total_n: num(v.planned_total_n, "Planned total"), n_looks: num(v.n_looks, "Planned checks"), spending: v.spending }),
    Result: SequentialResult,
  },
  {
    id: "peeking", group: "ab", label: "Why peeking is dangerous", blurb: "Simulation: no real effect, many checks.", needsData: false,
    fields: [
      { key: "n_looks", label: "Number of checks", type: "select", options: [["2", "2"], ["5", "5"], ["10", "10"]], default: "5" },
      { key: "n_sims", label: "Simulated experiments", type: "select", options: [["5000", "5,000"], ["20000", "20,000"], ["50000", "50,000"]], default: "20000" },
      alpha,
    ],
    run: (v) => api.post("/sequential/peeking-simulation", { n_looks: num(v.n_looks, "Checks"), n_sims: num(v.n_sims, "Simulations"), alpha: num(v.alpha, "Significance level") }),
    Result: PeekingResult,
  },
  {
    id: "psm", group: "causal", label: "Match similar people", blurb: "Propensity score matching, with balance checks.", needsData: true,
    fields: [treatCol, outcomeCol, covariates,
      { key: "caliper_sd", label: "Match strictness", type: "select", options: [["0.05", "Strict"], ["0.1", "Medium"], ["0.2", "Loose"]], default: "0.05",
        help: "Stricter matching balances the groups better but drops more treated people." },
      { key: "n_boot", label: "Interval accuracy", type: "select", options: [["100", "Quick"], ["200", "Standard"], ["500", "Precise"]], default: "200" }],
    async run(v, ds) {
      needCovs(v);
      const caliper_sd = num(v.caliper_sd, "Match strictness");
      const body = { ...causalBase(v, ds), caliper_sd, n_boot: num(v.n_boot, "Accuracy") };
      const [psm, sens] = await Promise.all([api.post("/causal/psm", body), api.post("/causal/psm-sensitivity", { ...causalBase(v, ds), caliper_sd }).catch(() => null)]);
      return { psm, sens };
    },
    Result: PsmResult,
  },
  {
    id: "did", group: "causal", label: "Before and after, versus a comparison group", blurb: "Difference-in-differences with a placebo check.", needsData: true,
    fields: [
      treatCol,
      { key: "pre_col", label: "Outcome before", type: "column", kind: "numeric", default: (d) => d.pre },
      { key: "post_col", label: "Outcome after", type: "column", kind: "numeric", default: (d) => d.post },
      { key: "placebo_pre_col", label: "Placebo: earlier outcome", help: "Optional. Two periods before treatment.", type: "column", kind: "numeric", optional: true, default: (d) => d.placeboPre },
      { key: "placebo_post_col", label: "Placebo: later outcome", type: "column", kind: "numeric", optional: true, default: (d) => d.placeboPost },
    ],
    async run(v, ds) {
      const withPlacebo = v.placebo_pre_col && v.placebo_post_col;
      const r = await api.post("/causal/did", { dataset_id: ds.dataset_id, treat_col: v.treat_col, pre_col: v.pre_col, post_col: v.post_col,
        placebo_pre_col: withPlacebo ? v.placebo_pre_col : null, placebo_post_col: withPlacebo ? v.placebo_post_col : null });
      return { ...r, pre_label: v.pre_col, post_label: v.post_col };
    },
    Result: DidResult,
  },
  {
    id: "dowhy", group: "causal", label: "Check it three ways and try to break it", blurb: "DoWhy estimates plus refutation tests.", needsData: true, slow: true,
    fields: [treatCol, outcomeCol, covariates, { key: "n_simulations", label: "Thoroughness", type: "select", options: [["10", "Quick"], ["25", "Standard"], ["50", "Thorough"]], default: "25" }],
    run: (v, ds) => { needCovs(v); return api.post("/causal/dowhy", { ...causalBase(v, ds), n_simulations: num(v.n_simulations, "Thoroughness") }); },
    Result: DowhyResult,
  },
  {
    id: "econml", group: "causal", label: "Machine-learning estimates", blurb: "EconML: double ML, causal forest, doubly robust.", needsData: true, slow: true,
    fields: [treatCol, outcomeCol, covariates, { key: "n_trees", label: "Forest size", type: "select", options: [["200", "Small (faster)"], ["500", "Standard"]], default: "200" }],
    run: (v, ds) => { needCovs(v); return api.post("/causal/econml", { ...causalBase(v, ds), n_trees: num(v.n_trees, "Forest size") }); },
    Result: EconmlResult,
  },
];

export function initialValues(method, defs, ds) {
  const out = {};
  for (const f of method.fields) out[f.key] = typeof f.default === "function" ? f.default(defs, ds) : f.default ?? (f.type === "columns" ? [] : "");
  return out;
}
