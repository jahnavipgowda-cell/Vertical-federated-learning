import { useState } from "react";
import { Loader2, Play, Trophy } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import { api } from "../api";
import { useSession } from "../SessionContext";

const COLORS = {
  "LOCAL": "#8CA0C2",
  "VFL (vanilla)": "#5A6B8C",
  "PCA-ROM only": "#F0B429",
  "VFedRS (full)": "#3FD9C7",
};

export default function ModelEvaluation() {
  const { status, refresh } = useSession();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const icgmReady = status?.steps_completed?.icgm;

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.runEvaluation();
      setResult(res);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (!status?.active) {
    return <PageHeader eyebrow="step 05" title="Model Evaluation" description="Create a session first on Party Setup." />;
  }
  if (!icgmReady) {
    return <PageHeader eyebrow="step 05" title="Model Evaluation" description="Train Representation Synthesis (ICGM) first." />;
  }

  const chartData = result ? Object.entries(result.accuracy).map(([name, value]) => ({ name, value })) : [];
  const winner = result ? Object.entries(result.accuracy).sort((a, b) => b[1] - a[1])[0] : null;

  return (
    <div>
      <PageHeader
        eyebrow="step 05"
        title="Model Evaluation"
        description="A downstream classifier is trained on each representation type and evaluated on a held-out test set that only exposes the active party's own features — exactly like the paper's evaluation protocol."
        right={
          <button onClick={run} disabled={loading} className="btn-primary inline-flex items-center gap-2">
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {loading ? "Training classifiers…" : "Run evaluation"}
          </button>
        }
      />

      {error && <div className="text-danger text-sm mb-6">{error}</div>}

      {result && (
        <>
          {winner && (
            <div className="panel p-5 mb-8 border-signal/30 bg-signal/5 flex items-center gap-3">
              <Trophy className="w-5 h-5 text-signal shrink-0" />
              <div className="text-sm">
                <span className="text-textPrimary font-medium">{winner[0]}</span>
                <span className="text-textMuted"> achieved the highest test accuracy at </span>
                <span className="font-mono text-signal">{winner[1]}%</span>
                <span className="text-textMuted"> on {result.n_aligned + result.n_nonaligned} training samples ({result.n_aligned} aligned).</span>
              </div>
            </div>
          )}

          <div className="panel p-6 mb-8">
            <div className="font-display font-medium mb-4">Test accuracy by method</div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#24314D" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} stroke="#8CA0C2" fontSize={11} unit="%" />
                <YAxis type="category" dataKey="name" stroke="#8CA0C2" fontSize={12} width={110} />
                <Tooltip contentStyle={{ background: "#121B2E", border: "1px solid #24314D", borderRadius: 8 }} labelStyle={{ color: "#E7EEF7" }} formatter={(v) => `${v}%`} />
                <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                  {chartData.map((d) => (
                    <Cell key={d.name} fill={COLORS[d.name] || "#3FD9C7"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard label="Representation MSE (aligned)" value={result.representation_mse_aligned} sub="synthetic vs. federated target" />
            <StatCard
              label="Communication reduction"
              value={`${result.communication_cost_estimate.speedup_factor}×`}
              sub="VFedRS vs. vanilla VFL"
              accent="mask"
            />
            <StatCard label="Sample split" value={`${result.n_aligned} / ${result.n_nonaligned}`} sub="aligned / non-aligned used in training" />
          </div>

          <div className="panel p-6 mt-8 text-sm text-textMuted leading-relaxed">
            <span className="text-textPrimary font-medium">Reading these results: </span>
            LOCAL and vanilla VFL both ignore most of what's available to the active party — LOCAL never
            sees passive features at all, and vanilla VFL throws away every non-aligned sample. PCA-ROM
            only and VFedRS both let non-aligned samples contribute; VFedRS additionally learns a direct
            mapping from the active party's raw features to the federated representation space, so it
            keeps working even when the aligned ratio is very low. Try lowering the aligned ratio on the
            Data Alignment page and re-running this pipeline to see the gap widen — this mirrors Fig. 5
            in the paper.
          </div>
        </>
      )}
    </div>
  );
}
