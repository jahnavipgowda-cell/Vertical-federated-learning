import { useMemo } from "react";

/**
 * The signature visual for VFedRS Studio: the active party (A) sits at the hub, passive
 * parties (P1..Pk) surround it. When `pulse` is true, small dots animate along the edges
 * to represent masked feature matrices flowing toward the active party during PCA-ROM.
 */
export default function NetworkDiagram({ nPassive = 3, pulse = false, highlight = "all", size = 280 }) {
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.34;

  const nodes = useMemo(() => {
    return Array.from({ length: nPassive }).map((_, i) => {
      const angle = (2 * Math.PI * i) / nPassive - Math.PI / 2;
      return {
        id: `P${i + 1}`,
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
      };
    });
  }, [nPassive, cx, cy, radius]);

  return (
    <svg viewBox={`0 0 ${size} ${size}`} width="100%" height="100%" className="overflow-visible">
      <defs>
        <radialGradient id="hubGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#3FD9C7" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#3FD9C7" stopOpacity="0" />
        </radialGradient>
      </defs>

      {nodes.map((n) => (
        <g key={n.id}>
          <line
            x1={cx}
            y1={cy}
            x2={n.x}
            y2={n.y}
            stroke="#24314D"
            strokeWidth="1.5"
            strokeDasharray={highlight === "all" || highlight === n.id ? "0" : "3 3"}
            opacity={highlight === "all" || highlight === n.id ? 1 : 0.35}
          />
          {pulse && (
            <circle r="3.2" fill="#F0B429">
              <animateMotion
                dur={`${1.6 + Math.random() * 0.8}s`}
                repeatCount="indefinite"
                path={`M${n.x},${n.y} L${cx},${cy}`}
              />
            </circle>
          )}
        </g>
      ))}

      <circle cx={cx} cy={cy} r={radius * 0.62} fill="url(#hubGlow)" />

      {nodes.map((n) => (
        <g key={`node-${n.id}`}>
          <circle cx={n.x} cy={n.y} r="20" fill="#182640" stroke="#24314D" strokeWidth="1.5" />
          <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize="11" fontFamily="'IBM Plex Mono', monospace" fill="#8CA0C2">
            {n.id}
          </text>
        </g>
      ))}

      <circle cx={cx} cy={cy} r="28" fill="#0B1220" stroke="#3FD9C7" strokeWidth="2" />
      <text x={cx} y={cy + 5} textAnchor="middle" fontSize="13" fontFamily="'Space Grotesk', sans-serif" fontWeight="600" fill="#3FD9C7">
        A
      </text>
    </svg>
  );
}
