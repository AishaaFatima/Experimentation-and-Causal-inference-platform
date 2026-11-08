import { useEffect, useMemo, useRef, useState } from "react";
import { api, BASE } from "./api";
import { defaultsFor, isNumeric } from "./lib";
import { GROUPS, METHODS, initialValues } from "./methods.jsx";
import { RawJson } from "./ui.jsx";

const SOURCES = [
  ["lalonde", "Job-training sample (LaLonde)"],
  ["simulate", "Simulated A/B test"],
  ["upload", "Upload a CSV file"],
];

/* ---------- left picker: where the data comes from ---------- */

function DataPicker({ dataset, onLoaded }) {
  const [source, setSource] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sim, setSim] = useState({ metric: "binary", n_per_group: "2000", lift: "0.02" });

  async function load(fn) {
    setBusy(true);
    setError("");
    try {
      onLoaded(await fn());
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  function choose(e) {
    const v = e.target.value;
    setSource(v);
    setError("");
    if (v === "lalonde") load(api.lalonde);
  }
  const simulate = () =>
    load(() => api.simulate({ metric: sim.metric, n_per_group: Number(sim.n_per_group), lift: Number(sim.lift), baseline_rate: 0.1, mean: 100, std: 20, seed: 1 }));

  return (
    <div className="picker">
      <label className="sr-only" htmlFor="data-source">Choose your data</label>
      <select id="data-source" value={source} onChange={choose}>
        <option value="" disabled>Choose your data</option>
        {SOURCES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>

      {source === "simulate" && (
        <div className="minform">
          <div className="field">
            <label htmlFor="sm">Outcome type</label>
            <select id="sm" value={sim.metric} onChange={(e) => setSim({ ...sim, metric: e.target.value, lift: e.target.value === "binary" ? "0.02" : "5" })}>
              <option value="binary">Rate (converted or not)</option>
              <option value="continuous">Average (for example revenue)</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="sn">People per group</label>
            <input id="sn" type="number" min="10" value={sim.n_per_group} onChange={(e) => setSim({ ...sim, n_per_group: e.target.value })} />
          </div>
          <div className="field">
            <label htmlFor="sl">True effect of the treatment</label>
            <input id="sl" type="number" step="any" value={sim.lift} onChange={(e) => setSim({ ...sim, lift: e.target.value })} />
            <small>{sim.metric === "binary" ? "0.02 raises a 10% rate to 12%." : "Added to an average of 100."}</small>
          </div>
          <button className="btn primary" disabled={busy} onClick={simulate}>{busy ? "Generating…" : "Generate data"}</button>
        </div>
      )}
      {source === "upload" && (
        <div className="minform">
          <div className="field">
            <label htmlFor="csv">CSV file</label>
            <input id="csv" type="file" accept=".csv,text/csv" disabled={busy} onChange={(e) => { const f = e.target.files[0]; if (f) load(() => api.upload(f)); }} />
            <small>One row per person, with a column for the group and one for the outcome.</small>
          </div>
        </div>
      )}
      {busy && source === "lalonde" && <p className="hint" role="status">Loading…</p>}
      {error && <div className="error" role="alert"><b>Couldn't load that data</b>{error}</div>}
      {dataset && !error && (
        <div className="dataset">
          <p><b>{dataset.name}</b>: {dataset.n_rows.toLocaleString()} rows, {dataset.columns.length} columns</p>
          {dataset.notes && <p>{dataset.notes}</p>}
          {dataset.true_effect !== null && dataset.true_effect !== undefined && <p>The true effect built into this data is <b>{dataset.true_effect}</b>. A good method should land near it.</p>}
          <details>
            <summary>Preview the data</summary>
            <div className="preview">
              <table>
                <thead><tr>{dataset.columns.slice(0, 8).map((c) => <th key={c.name}>{c.name}</th>)}</tr></thead>
                <tbody>{dataset.preview.map((row, i) => <tr key={i}>{dataset.columns.slice(0, 8).map((c) => <td key={c.name}>{String(row[c.name] ?? "")}</td>)}</tr>)}</tbody>
              </table>
            </div>
          </details>
        </div>
      )}
    </div>
  );
}

/* ---------- right picker: what to do ---------- */

function MethodPicker({ method, onPick }) {
  return (
    <div className="picker">
      <label className="sr-only" htmlFor="method">What do you want to do?</label>
      <select id="method" value={method ? method.id : ""} onChange={(e) => onPick(e.target.value)}>
        <option value="" disabled>Choose what to do</option>
        {GROUPS.map((g) => (
          <optgroup key={g.id} label={g.title}>
            {METHODS.filter((m) => m.group === g.id).map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </optgroup>
        ))}
      </select>
      {method && <p className="hint">{method.blurb}{method.needsData ? " Needs data." : ""}</p>}
    </div>
  );
}

/* ---------- set it up ---------- */

function Field({ f, values, set, dataset }) {
  const cols = dataset ? dataset.columns : [];
  const id = `f-${f.key}`;
  const value = values[f.key];
  const help = typeof f.help === "function" ? f.help(values) : f.help;
  let control;
  if (f.type === "select") {
    control = <select id={id} value={value} onChange={(e) => set(f.key, e.target.value)}>{f.options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>;
  } else if (f.type === "column") {
    const options = f.kind === "numeric" ? cols.filter(isNumeric) : cols;
    control = (
      <select id={id} value={value} onChange={(e) => set(f.key, e.target.value)}>
        {f.optional && <option value="">None</option>}
        {options.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
      </select>
    );
  } else if (f.type === "columns") {
    const taken = [values.treat_col, values.group_col, values.outcome_col];
    const options = cols.filter((c) => isNumeric(c) && !taken.includes(c.name));
    control = (
      <div className="chips" role="group" aria-labelledby={`${id}-l`}>
        {options.map((c) => (
          <button type="button" key={c.name} className="chip" aria-pressed={value.includes(c.name)}
            onClick={() => set(f.key, value.includes(c.name) ? value.filter((n) => n !== c.name) : [...value, c.name])}>{c.name}</button>
        ))}
      </div>
    );
  } else {
    control = <input id={id} type={f.type === "number" ? "number" : "text"} step={f.step} value={value} onChange={(e) => set(f.key, e.target.value)} />;
  }
  return (
    <div className={`field${f.type === "columns" ? " full" : f.wide ? " wide" : ""}`}>
      {f.type === "columns" ? <span className="lab" id={`${id}-l`}>{f.label}</span> : <label htmlFor={id}>{f.label}</label>}
      {control}
      {help && <small>{help}</small>}
    </div>
  );
}

function Setup({ method, values, set, dataset, onRun, status }) {
  const blocked = method.needsData && !dataset;
  return (
    <section className="setup" aria-label="Set it up">
      <h2>Set it up</h2>
      {blocked ? (
        <p className="hint blocked">This analysis needs data. Choose your data above first.</p>
      ) : (
        <>
          <div className="fields">
            {method.fields.filter((f) => !f.showIf || f.showIf(values)).map((f) => <Field key={f.key} f={f} values={values} set={set} dataset={dataset} />)}
          </div>
          <div className="runrow">
            <button className="btn primary run" onClick={onRun} disabled={status === "running"}>{status === "running" ? "Running…" : "Run analysis"}</button>
            {method.slow && <p className="hint">This one can take up to a minute.</p>}
          </div>
        </>
      )}
    </section>
  );
}

/* ---------- result ---------- */

function Results({ method, dataset, state }) {
  if (state.status === "idle") return null;
  if (state.status === "running") return <p className="busy" role="status">Running {method.label.toLowerCase()}…{method.slow ? " This can take up to a minute." : ""}</p>;
  if (state.status === "error") return <div className="error" role="alert"><b>That didn't work</b>{state.error}</div>;
  const { Result } = method;
  return (
    <>
      <p className="rhead">{method.label}{method.needsData && dataset ? ` · ${dataset.name}` : ""}</p>
      <div className="result"><Result r={state.result} /></div>
      <RawJson data={state.result} />
    </>
  );
}

function Welcome() {
  return (
    <div className="empty">
      <h2>Pick your data and what you want to do.</h2>
      <p>You'll get a plain-English conclusion first, then the numbers and a chart behind it.</p>
      <p>A good first try: choose the job-training sample, then "Compare averages". The raw comparison suggests the program <em>lowered</em> earnings. Then try "Match similar people" and see what changes.</p>
    </div>
  );
}

/* ---------- app ---------- */

export default function App() {
  const [health, setHealth] = useState("checking");
  const [dataset, setDataset] = useState(null);
  const [methodId, setMethodId] = useState("");
  const method = useMemo(() => METHODS.find((m) => m.id === methodId) || null, [methodId]);
  const defs = useMemo(() => defaultsFor(dataset ? dataset.columns : []), [dataset]);
  const stageRef = useRef(null);

  // Form values and results belong to one (method, dataset) pair. Deriving them from that key, instead of
  // resetting them in an effect, means a screen is never drawn with another method's values or result.
  const ctx = `${methodId}|${dataset ? dataset.dataset_id : ""}`;
  const [form, setForm] = useState({ ctx: "", values: {} });
  const [runState, setRun] = useState({ ctx: "", status: "idle" });
  const values = !method ? {} : form.ctx === ctx ? form.values : initialValues(method, defs, dataset);
  const state = runState.ctx === ctx ? runState : { status: "idle" };

  const checkHealth = () => { setHealth("checking"); api.health().then(() => setHealth("ok")).catch(() => setHealth("off")); };
  useEffect(checkHealth, []);

  // When a run starts, bring the result into view if it is off screen.
  useEffect(() => {
    const el = stageRef.current;
    if (state.status !== "running" || !el) return;
    const top = el.getBoundingClientRect().top;
    if (top < 0 || top > window.innerHeight - 140) {
      const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      el.scrollIntoView?.({ block: "start", behavior: calm ? "auto" : "smooth" });
    }
  }, [state.status]);

  const set = (key, val) => setForm({ ctx, values: { ...values, ...(method.onChange ? method.onChange(key, val) : {}), [key]: val } });

  async function run() {
    setRun({ ctx, status: "running" });
    try {
      setRun({ ctx, status: "done", result: await method.run(values, dataset) });
    } catch (e) {
      setRun({ ctx, status: "error", error: e.message });
    }
  }

  return (
    <div className="app">
      <button className="status" onClick={checkHealth} title="Check the connection again">
        <span className={`dot ${health === "ok" ? "ok" : health === "off" ? "off" : ""}`} />
        {health === "ok" ? "Backend connected" : health === "off" ? `Backend not found at ${BASE}. Click to retry.` : "Connecting…"}
      </button>
      <header className="top">
        <h1>It's that tool</h1>
        <p className="tagline">Plan <span aria-hidden="true">•</span> Analyze <span aria-hidden="true">•</span> Estimate</p>
      </header>

      <div className="pickers">
        <DataPicker dataset={dataset} onLoaded={setDataset} />
        <MethodPicker method={method} onPick={setMethodId} />
      </div>

      <div className="workspace">
        {!method ? (
          <Welcome />
        ) : (
          <>
            <Setup method={method} values={values} set={set} dataset={dataset} onRun={run} status={state.status} />
            <div className="results" ref={stageRef} aria-live="polite"><Results method={method} dataset={dataset} state={state} /></div>
          </>
        )}
      </div>
    </div>
  );
}
