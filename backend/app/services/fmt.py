"""Number formatting for the plain-English summaries."""
import math


def fmt_num(v: float, sig: int = 4, signed: bool = False, ref: float | None = None) -> str:
    """Round to `sig` significant figures with thousands separators: 6349.14 -> 6,349, 58.123 -> 58.12.
    Pass `ref` (the largest number in the same sentence) so all of them share one number of decimals."""
    if v == 0 or v is None or (isinstance(v, float) and math.isnan(v)):
        return "0" if v == 0 else "–"
    scale = abs(ref) if ref else abs(v)
    digits = max(sig - int(math.floor(math.log10(scale))) - 1, 0)
    out = f"{round(v, digits):,.{digits}f}"
    return f"+{out}" if signed and v > 0 else out
