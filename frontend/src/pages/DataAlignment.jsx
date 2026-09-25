import { useEffect, useState } from "react";
import { Loader2, RefreshCw, Lock, Unlock } from "lucide-react";
import PageHeader from "../components/PageHeader";
import { api } from "../api";
import { useSession } from "../SessionContext";

export default function DataAlignment() {
  const { status, refresh } = useSession();
  const [ratio, setRatio] = useState(0.35);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadPreview = async () => {
    try {
      const p = await api.psiPreview();
      setPreview(p);
      setRatio(p.aligned_ratio);
    } catch (e) {
      setError(e.message);
    }
  };

  useEffect(() => {
    if (status?.active) loadPreview();
  }, [status?.active]);

  const rerun = async () => {
    setLoading(true);
    setError(null);
    try {
      await api.psiAlign({ aligned_ratio: ratio });
      await refresh();
      await loadPreview();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (!status?.active) {
    return (
      <div>
        <PageHeader eyebrow="step 02" title="Data Alignment (PSI)" description="Create a session first on Party Setup." />
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        eyebrow="step 02"
        title="Data Alignment (PSI)"
        description="Parties intersect sample IDs via private set intersection. Only aligned samples are visible to every party — the active party keeps the rest entirely to itself."
      />

      <div className="panel p-6 mb-8">
        <span className="field-label">Aligned sample ratio ({Math.round(ratio * 100)}%)</span>
        <div className="flex items-center gap-4">
          <input type="range" min="0.05" max="0.85" step="0.05" value={ratio} onChange={(e) => setRatio(Number(e.target.value))} className="flex-1 accent-mask" />
          <button onClick={rerun} disabled={loading} className="btn-secondary inline-flex items-center gap-2 whitespace-nowrap">
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
            Re-run PSI
          </button>
        </div>
        {error && <div className="text-danger text-sm mt-3">{error}</div>}
      </div>

      {preview && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="panel p-6">
            <div className="flex items-center gap-2 mb-4">
              <Unlock className="w-4 h-4 text-signal" />
              <div className="font-display font-medium">Aligned samples</div>
              <span className="ml-auto text-xs font-mono text-textMuted">{status.n_aligned} total</span>
            </div>
            <div className="space-y-2">
              {preview.aligned_sample_preview.map((s) => (
                <div key={s.sample_id} className="flex items-center justify-between text-sm bg-panelAlt rounded-lg px-3 py-2">
                  <span className="font-mono text-textMuted">#{s.sample_id}</span>
                  <div className="flex gap-1">
                    {s.parties_with_access.map((p) => (
                      <span key={p} className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${p === "A" ? "bg-signal/20 text-signal" : "bg-line text-textMuted"}`}>{p}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel p-6">
            <div className="flex items-center gap-2 mb-4">
              <Lock className="w-4 h-4 text-mask" />
              <div className="font-display font-medium">Non-aligned samples</div>
              <span className="ml-auto text-xs font-mono text-textMuted">{status.n_nonaligned} total</span>
            </div>
            <div className="space-y-2">
              {preview.nonaligned_sample_preview.map((s) => (
                <div key={s.sample_id} className="flex items-center justify-between text-sm bg-panelAlt rounded-lg px-3 py-2">
                  <span className="font-mono text-textMuted">#{s.sample_id}</span>
                  <div className="flex gap-1">
                    {s.parties_with_access.map((p) => (
                      <span key={p} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-mask/20 text-mask">{p}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-textMuted mt-4 leading-relaxed">
              These samples never leave the active party and traditionally go completely unused by
              vanilla VFL. VFedRS is what lets them contribute to training.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
