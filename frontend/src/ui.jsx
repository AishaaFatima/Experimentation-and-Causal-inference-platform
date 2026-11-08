import { fmtNum, niceTicks, wrapLabel } from "./lib";

/** The plain-English answer. This is the page's main event, so it is set large. */
export function Conclusion({ children }) {
  return <p className="conclusion">{children}</p>;
}

export function Stats({ items }) {
  return (
    <dl className="stats">
      {items.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Notes({ items, tone = "warn" }) {
  const list = (items || []).filter(Boolean);
  if (!list.length) return null;
  return (
    <ul className={`notes ${tone}`}>
      {list.map((t) => (
        <li key={t}>{t}</li>
      ))}
    </ul>
  );
}

export function ChartBox({ title, caption, children }) {
  return (
    <figure className="chartbox">
      <figcaption className="charttitle">{title}</figcaption>
      {children}
      {caption && <p className="chartcaption">{caption}</p>}
    </figure>
  );
}

export function Table({ cols, rows }) {
  return (
    <div className="tablewrap">
      <table>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c.key} className={c.num ? "num" : ""}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c.key} className={c.num ? "num" : ""}>
                  {c.render ? c.render(r) : r[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Badge({ tone, children }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

export function RawJson({ data }) {
  return (
    <details className="raw">
      <summary>Show the raw response</summary>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}

/**
 * Forest plot: each row is an estimate with an optional confidence interval, drawn against "no effect" (zero).
 * rows: [{ label, est, lo?, hi?, tone?: "main" | "naive" | "muted" }]
 */
export function ForestPlot({ rows, format = fmtNum, label = "Estimates with confidence intervals" }) {
  const W = 572, left = 206, right = 28, top = 24, rowH = 46, axisH = 34;
  const H = top + rows.length * rowH + axisH;
  const vals = rows.flatMap((r) => [r.est, r.lo, r.hi]).filter((v) => Number.isFinite(v)).concat(0);
  let min = Math.min(...vals);
  let max = Math.max(...vals);
  const pad = (max - min) * 0.08 || 1;
  min -= pad;
  max += pad;
  const x = (v) => left + ((v - min) / (max - min)) * (W - left - right);
  return (
    <svg className="forest" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      {niceTicks(min, max).map((t) => (
        <g key={t}>
          <line className="grid" x1={x(t)} x2={x(t)} y1={top - 8} y2={H - axisH} />
          <text className="tick" x={x(t)} y={H - axisH + 20} textAnchor="middle">
            {format(t)}
          </text>
        </g>
      ))}
      <line className="zero" x1={x(0)} x2={x(0)} y1={top - 8} y2={H - axisH} />
      {rows.map((r, i) => {
        const y = top + i * rowH + rowH / 2 + 4;
        const hasCI = Number.isFinite(r.lo) && Number.isFinite(r.hi);
        return (
          <g key={r.label} className={`frow ${r.tone || "main"}`}>
            <text className="flabel" x={left - 14} y={y + 4} textAnchor="end">
              <title>{r.label}</title>
              {wrapLabel(r.label, 23).map((ln, k, all) => (
                <tspan key={k} x={left - 14} dy={k === 0 ? (all.length > 1 ? -7 : 0) : 15}>{ln}</tspan>
              ))}
            </text>
            {hasCI && (
              <>
                <line className="ci" x1={x(r.lo)} x2={x(r.hi)} y1={y} y2={y} />
                <line className="cap" x1={x(r.lo)} x2={x(r.lo)} y1={y - 5} y2={y + 5} />
                <line className="cap" x1={x(r.hi)} x2={x(r.hi)} y1={y - 5} y2={y + 5} />
              </>
            )}
            <circle cx={x(r.est)} cy={y} r="5.5" />
            <text className="fval" x={x(r.est)} y={y - 11} textAnchor="middle">
              {format(r.est)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
