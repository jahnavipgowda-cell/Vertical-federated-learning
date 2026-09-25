import { useEffect, useState } from "react";
import { Play, Loader2 } from "lucide-react";
import PageHeader from "../components/PageHeader";
import NetworkDiagram from "../components/NetworkDiagram";
import { api } from "../api";
import { useSession } from "../SessionContext";

const DEFAULTS = {
  preset: "clinical",
  n_samples: 1200,
  n_passive: 3,
  active_frac: 0.2,
  aligned_ratio: 0.35,
  seed: 42,
};

export default function PartySetup() {
  const { status, refresh } = useSession();
  const [presets, setPresets] = useState({});
  const [form, setForm] = useState(DEFAULTS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    api.presets().then(setPresets).catch(() => {});
  }, []);

  useEffect(() => {
    if (status?.active && status.config) {
      setForm((f) => ({ ...f, ...status.config }));
    }
  }, [status]);

  const update = (key) => (e) => {
    const val = e.target.type === "range" || e.target.type === "number" ? Number(e.target.value) : e.target.value;
    setForm((f) => ({ ...f, [key]: val }));
  };

  const submit = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.newSession(form);
      setResult(res);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const activePreset = presets[form.preset];

  return (
    <div>
      <PageHeader
        eyebrow="step 01"
        title="Party Setup"
        description="Pick a dataset and decide how its features are vertically split between the active party (who holds the labels) and K passive parties."
      />

      <div className="grid grid-cols-1 lg:grid-cols-[1.1fr_0.9fr] gap-8">
        <div className="panel p-6 space-y-6">
          <div>
            <span className="field-label">Dataset</span>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(presets).map(([key, p]) => (
                <button
                  key={key}
                  onClick={() => setForm((f) => ({ ...f, preset: key }))}
                  className={`text-left p-3 rounded-lg border transition ${
                    form.preset === key ? "border-signal bg-signal/10" : "border-line bg-panelAlt hover:border-signal/40"
                  }`}
                >
                  <div className="text-sm font-medium text-textPrimary">{p.label}</div>
                  <div className="text-xs text-textMuted mt-1">{p.n_features} features · {p.n_classes} classes</div>
                </button>
              ))}
            </div>
            {activePreset && <p className="text-xs text-textMuted mt-2">{activePreset.description}</p>}
          </div>

          <div>
            <span className="field-label">Number of passive parties (K = {form.n_passive})</span>
            <input type="range" min="1" max="5" step="1" value={form.n_passive} onChange={update("n_passive")} className="w-full accent-signal" />
          </div>

          <div>
            <span className="field-label">Active party's share of features ({Math.round(form.active_frac * 100)}%)</span>
            <input type="range" min="0.1" max="0.6" step="0.05" value={form.active_frac} onChange={update("active_frac")} className="w-full accent-signal" />
            <p className="text-xs text-textMuted mt-1">Lower values mean the active party depends more on passive parties' knowledge.</p>
          </div>

          <div>
            <span className="field-label">Aligned sample ratio ({Math.round(form.aligned_ratio * 100)}%)</span>
            <input type="range" min="0.05" max="0.8" step="0.05" value={form.aligned_ratio} onChange={update("aligned_ratio")} className="w-full accent-mask" />
            <p className="text-xs text-textMuted mt-1">Fraction of training samples visible to every party via PSI. The rest stay local to the active party.</p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="field-label">Training samples</span>
              <input type="number" min="200" max="6000" step="100" value={form.n_samples} onChange={update("n_samples")} className="w-full bg-panelAlt border border-line rounded-lg px-3 py-2 text-sm font-mono focus:border-signal outline-none" />
            </div>
            <div>
              <span className="field-label">Random seed</span>
              <input type="number" value={form.seed} onChange={update("seed")} className="w-full bg-panelAlt border border-line rounded-lg px-3 py-2 text-sm font-mono focus:border-signal outline-none" />
            </div>
          </div>

          {error && <div className="text-danger text-sm bg-danger/10 border border-danger/30 rounded-lg px-3 py-2">{error}</div>}

          <button onClick={submit} disabled={loading} className="btn-primary w-full inline-flex items-center justify-center gap-2">
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {loading ? "Generating & partitioning data…" : "Create session"}
          </button>
        </div>

        <div className="panel p-6 flex flex-col items-center justify-center">
          <NetworkDiagram nPassive={form.n_passive} pulse={loading} size={220} />
          {result && (
            <div className="w-full mt-6 space-y-2">
              <div className="text-xs uppercase tracking-wider text-textMuted font-mono mb-2">Feature partition</div>
              {Object.entries(result.party_feature_counts).map(([party, n]) => (
                <div key={party} className="flex items-center justify-between text-sm">
                  <span className={party === "A" ? "text-signal font-medium" : "text-textMuted"}>{party}{party === "A" ? " (active)" : ""}</span>
                  <span className="font-mono text-textPrimary">{n} features</span>
                </div>
              ))}
              <div className="pt-3 mt-3 border-t border-line flex justify-between text-sm">
                <span className="text-textMuted">Aligned / non-aligned</span>
                <span className="font-mono text-textPrimary">{result.n_aligned} / {result.n_nonaligned}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
