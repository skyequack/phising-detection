type Props = { probability: number; threshold: number };

// Half-circle meter: 0% at the left, 100% at the right, with the decision threshold marked.
const R = 90;
const CX = 110;
const CY = 110;

function point(frac: number, r = R) {
  const a = Math.PI * (1 - frac);
  return { x: CX + r * Math.cos(a), y: CY - r * Math.sin(a) };
}

function arc(from: number, to: number) {
  const a = point(from);
  const b = point(to);
  return `M ${a.x} ${a.y} A ${R} ${R} 0 0 1 ${b.x} ${b.y}`;
}

export default function Gauge({ probability, threshold }: Props) {
  const phishing = probability >= threshold;
  const color = phishing ? "var(--danger)" : "var(--safe)";
  const tickIn = point(threshold, R - 14);
  const tickOut = point(threshold, R + 10);
  const label = point(threshold, R + 22);
  const needle = point(probability, R - 8);

  return (
    <figure className="m-0">
      <svg viewBox="0 0 220 135" role="img" className="w-full max-w-xs"
        aria-label={`Model score ${(probability * 100).toFixed(1)} percent, decision threshold ${(threshold * 100).toFixed(0)} percent`}>
        <path d={arc(0, 1)} fill="none" stroke="var(--panel-border)" strokeWidth="14" strokeLinecap="round" />
        {probability > 0.005 && (
          <path d={arc(0, Math.max(probability, 0.01))} fill="none" stroke={color} strokeWidth="14" strokeLinecap="round" />
        )}
        <line x1={tickIn.x} y1={tickIn.y} x2={tickOut.x} y2={tickOut.y} stroke="var(--foreground)" strokeWidth="2" />
        <text x={label.x} y={label.y} textAnchor="middle" fontSize="9" fill="var(--muted)">
          threshold
        </text>
        <circle cx={needle.x} cy={needle.y} r="4" fill="var(--foreground)" />
        <text x={CX} y={CY - 8} textAnchor="middle" fontSize="30" fontWeight="700" fill={color} className="font-mono">
          {(probability * 100).toFixed(1)}%
        </text>
        <text x={CX} y={CY + 10} textAnchor="middle" fontSize="10" fill="var(--muted)">
          model score
        </text>
      </svg>
    </figure>
  );
}
