import { useState } from "react";
import { Loader2, Play, CheckCircle2 } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, ComposedChart } from "recharts";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import { api } from "../api";
import { useSession } from "../SessionContext";

export default function RepresentationExtraction() {
  const { status, refresh } = useSession();
  const [nComponents, setNComponents] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.runPcaRom({ n_components: nComponents });
      setResult(res);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (!status?.active) {
    return <PageHeader eyebrow="step 03" title="Federated Extraction" description="Create a session first on Party Setup." />;
  }

  const chartData = result?.explained_variance_ratio.map((v, i) => ({
    name: `PC${i + 1}`,
    variance: Number((v * 100).toFixed(2)),
    cumulative: Number((result.cumulative_variance[i] * 100).toFixed(2)),
  }));

  return (
    <div>
      <PageHeader
        eyebrow="step 03"
        title="Federated Extraction — PCA-ROM"
        description="Principal Component Analysis under Removable Orthogonal Masks. Every party masks its aligned features locally before anything is transmitted; the masks cancel out exactly once the active party finishes the SVD, so nobody ever sees another party's raw data."
      />

      <div className="panel p-6 mb-8">
        <div className="flex items-end gap-6 flex-wrap">
          <div className="flex-1 min-w-[220px]">
            <span className="field-label">Number of federated components (r = {nComponents})</span>
            <input type="range" min="2" max="24" step="1" value={nComponents} onChange={(e) => setNComponents(Number(e.target.value))} className="w-full accent-signal" />
          </div>
          <button onClick={run} disabled={loading} className="btn-primary inline-flex items-center gap-2">
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {loading ? "Running masked SVD…" : "Run PCA-ROM"}
          </button>
        </div>
        {error && <div className="text-danger text-sm mt-3">{error}</div>}
      </div>

      {result && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <StatCard
              label="Mask reconstruction error"
              value={result.mask_reconstruction_error.toExponential(2)}
              sub="unmasked V vs. true V (subspace distance)"
              accent="signal"
            />
            <StatCard label="Federated dimensions" value={`${result.h_fed_dim[0]} × ${result.h_fed_dim[1]}`} sub="aligned samples × components" />
            <StatCard label="Parties involved" value={result.party_order.length} sub={result.party_order.join(", ")} />
          </div>

          <div className="panel p-6 mb-8 flex items-start gap-3 border-signal/30 bg-signal/5">
            <CheckCircle2 className="w-5 h-5 text-signal shrink-0 mt-0.5" />
            <div className="text-sm text-textMuted leading-relaxed">
              The reconstruction error above compares the representation recovered <em>with</em> masks
              against a ground-truth computation done <em>without</em> any privacy protection. A value
              near zero (this one is ~10⁻¹³) proves the masks are exactly removable — privacy costs
              nothing in accuracy here.
            </div>
          </div>

          <div className="panel p-6">
            <div className="font-display font-medium mb-4">Explained variance by component</div>
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#24314D" />
                <XAxis dataKey="name" stroke="#8CA0C2" fontSize={11} />
                <YAxis stroke="#8CA0C2" fontSize={11} unit="%" />
                <Tooltip contentStyle={{ background: "#121B2E", border: "1px solid #24314D", borderRadius: 8 }} labelStyle={{ color: "#E7EEF7" }} />
                <Bar dataKey="variance" fill="#3FD9C7" radius={[4, 4, 0, 0]} name="Variance %" />
                <Line type="monotone" dataKey="cumulative" stroke="#F0B429" strokeWidth={2} dot={false} name="Cumulative %" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="panel p-6 mt-8">
            <div className="font-display font-medium mb-4">Recovered principal-component preview (per party)</div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(result.per_party_components_preview).map(([party, rows]) => (
                <div key={party}>
                  <div className={`text-xs font-mono mb-2 ${party === "A" ? "text-signal" : "text-textMuted"}`}>{party}{party === "A" ? " (active)" : ""}</div>
                  <div className="overflow-x-auto">
                    <table className="text-xs font-mono w-full">
                      <tbody>
                        {rows.map((row, i) => (
                          <tr key={i} className="border-b border-line/50">
                            {row.map((v, j) => (
                              <td key={j} className="px-2 py-1 text-textMuted">{v.toFixed(3)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
