import { useState } from "react";
import { Loader2, Play } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import { api } from "../api";
import { useSession } from "../SessionContext";

export default function RepresentationSynthesis() {
  const { status, refresh } = useSession();
  const [form, setForm] = useState({ epochs: 100, lr_g: 0.015, lr_d: 0.01, lr_theta: 0.03, use_adv: true, use_gr: true, disc_every: 3 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const pcaReady = status?.steps_completed?.pca_rom;

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.trainIcgm(form);
      setResult(res);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (!status?.active) {
    return <PageHeader eyebrow="step 04" title="Representation Synthesis" description="Create a session first on Party Setup." />;
  }
  if (!pcaReady) {
    return <PageHeader eyebrow="step 04" title="Representation Synthesis" description="Run Federated Extraction (PCA-ROM) first." />;
  }

  const chartData = result?.history.epoch.map((e, i) => ({
    epoch: e,
    l_adv: result.history.l_adv[i],
    l_gr: result.history.l_gr[i],
    mse_aligned: result.history.mse_aligned[i],
  }));

  return (
    <div>
      <PageHeader
        eyebrow="step 04"
        title="Representation Synthesis — ICGM"
        description="The Input-Conditional Generative Model learns to turn the active party's own raw features into representations that look like the federated ones. Trained with an adversarial loss (aligned samples only) and a generative-regression loss (all samples, aligned and non-aligned alike)."
      />

      <div className="panel p-6 mb-8 space-y-5">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <span className="field-label">Training epochs ({form.epochs})</span>
            <input type="range" min="10" max="250" step="10" value={form.epochs} onChange={(e) => setForm((f) => ({ ...f, epochs: Number(e.target.value) }))} className="w-full accent-signal" />
          </div>
          <div>
            <span className="field-label">Generator learning rate ({form.lr_g})</span>
            <input type="range" min="0.002" max="0.05" step="0.002" value={form.lr_g} onChange={(e) => setForm((f) => ({ ...f, lr_g: Number(e.target.value) }))} className="w-full accent-signal" />
          </div>
          <div>
            <span className="field-label">Discriminator update every ({form.disc_every} epochs)</span>
            <input type="range" min="1" max="10" step="1" value={form.disc_every} onChange={(e) => setForm((f) => ({ ...f, disc_every: Number(e.target.value) }))} className="w-full accent-signal" />
          </div>
        </div>

        <div className="flex gap-6 pt-2 border-t border-line">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input type="checkbox" checked={form.use_adv} onChange={(e) => setForm((f) => ({ ...f, use_adv: e.target.checked }))} className="accent-signal w-4 h-4" />
            <span>Adversarial loss <span className="text-textMuted font-mono">(l_adv)</span></span>
          </label>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input type="checkbox" checked={form.use_gr} onChange={(e) => setForm((f) => ({ ...f, use_gr: e.target.checked }))} className="accent-mask w-4 h-4" />
            <span>Generative-regression loss <span className="text-textMuted font-mono">(l_gr)</span></span>
          </label>
          <p className="text-xs text-textMuted ml-auto self-center max-w-xs">
            Toggle these to reproduce the paper's ablation study (Section V-B6) — try running with only one loss active.
          </p>
        </div>

        <button onClick={run} disabled={loading} className="btn-primary w-full inline-flex items-center justify-center gap-2">
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          {loading ? "Training generator + discriminator…" : "Train ICGM"}
        </button>
        {error && <div className="text-danger text-sm">{error}</div>}
      </div>

      {result && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <StatCard label="Final MSE / dim (aligned)" value={result.final_mse_per_dim} sub="synthetic vs. true federated representation" accent="signal" />
            <StatCard label="Epochs trained" value={result.config.epochs} />
            <StatCard label="Losses active" value={[form.use_adv && "adv", form.use_gr && "gr"].filter(Boolean).join(" + ") || "none"} accent="mask" />
          </div>

          <div className="panel p-6">
            <div className="font-display font-medium mb-4">Training curves</div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#24314D" />
                <XAxis dataKey="epoch" stroke="#8CA0C2" fontSize={11} />
                <YAxis stroke="#8CA0C2" fontSize={11} />
                <Tooltip contentStyle={{ background: "#121B2E", border: "1px solid #24314D", borderRadius: 8 }} labelStyle={{ color: "#E7EEF7" }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="mse_aligned" stroke="#3FD9C7" strokeWidth={2} dot={false} name="MSE (aligned reps)" />
                <Line type="monotone" dataKey="l_adv" stroke="#F0B429" strokeWidth={2} dot={false} name="Adversarial loss" />
                <Line type="monotone" dataKey="l_gr" stroke="#F0654F" strokeWidth={2} dot={false} name="Generative-regression loss" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
