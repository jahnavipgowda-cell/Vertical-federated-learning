export default function StatCard({ label, value, sub, accent = "signal" }) {
  const accentClass = accent === "mask" ? "text-mask" : accent === "danger" ? "text-danger" : "text-signal";
  return (
    <div className="panel px-5 py-4">
      <div className="text-xs uppercase tracking-wider text-textMuted font-mono">{label}</div>
      <div className={`text-2xl font-display font-semibold mt-1 ${accentClass}`}>{value}</div>
      {sub && <div className="text-xs text-textMuted mt-1">{sub}</div>}
    </div>
  );
}
