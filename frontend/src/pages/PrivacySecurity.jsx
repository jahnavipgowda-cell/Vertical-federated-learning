import { useEffect, useState } from "react";
import { ShieldCheck, Lock, Eye, EyeOff, Radio } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import { api } from "../api";
import { useSession } from "../SessionContext";

export default function PrivacySecurity() {
  const { status } = useSession();
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (status?.active) {
      api.privacyReport().then(setReport).catch((e) => setError(e.message));
    }
  }, [status?.active]);

  if (!status?.active) {
    return <PageHeader eyebrow="privacy" title="Privacy & Security" description="Create a session first on Party Setup." />;
  }

  return (
    <div>
      <PageHeader
        eyebrow="privacy"
        title="Privacy & Security"
        description="What VFedRS's PCA-ROM protocol actually protects, under a semi-honest threat model — and the numeric proof that the removable masks don't distort the result."
      />

      {error && <div className="text-danger text-sm mb-6">{error}</div>}

      {report && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <StatCard
              label="Mask reconstruction error"
              value={report.mask_reconstruction_error !== null ? report.mask_reconstruction_error.toExponential(2) : "run PCA-ROM first"}
              sub="deviation from the non-private ground truth"
              accent="signal"
            />
            <StatCard label="Parties in this session" value={Object.keys(report.party_feature_counts).length} sub={Object.entries(report.party_feature_counts).map(([k, v]) => `${k}:${v}`).join("  ")} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            {report.assumptions.map((a) => (
              <div key={a.entity} className="panel p-5">
                <div className="flex items-center gap-2 mb-2">
                  <ShieldCheck className="w-4 h-4 text-signal" />
                  <div className="font-display font-medium text-sm">{a.entity}</div>
                </div>
                <p className="text-xs text-textMuted leading-relaxed">{a.assumption}</p>
              </div>
            ))}
          </div>

          <div className="panel p-6 mb-8">
            <div className="flex items-center gap-2 mb-4">
              <Radio className="w-4 h-4 text-mask" />
              <div className="font-display font-medium">Protocol rounds (PCA-ROM)</div>
            </div>
            <ol className="space-y-3">
              {report.protocol_rounds.map((r, i) => (
                <li key={i} className="flex gap-3 text-sm">
                  <span className="font-mono text-mask shrink-0">{String(i + 1).padStart(2, "0")}</span>
                  <span className="text-textMuted">{r}</span>
                </li>
              ))}
            </ol>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="panel p-6">
              <div className="flex items-center gap-2 mb-3">
                <EyeOff className="w-4 h-4 text-signal" />
                <div className="font-display font-medium text-sm">Active party can send</div>
              </div>
              <p className="text-xs text-textMuted leading-relaxed font-mono bg-panelAlt rounded-lg p-3">
                {report.what_leaves_each_party["Active Party"]}
              </p>
            </div>
            <div className="panel p-6">
              <div className="flex items-center gap-2 mb-3">
                <EyeOff className="w-4 h-4 text-mask" />
                <div className="font-display font-medium text-sm">Passive parties can send</div>
              </div>
              <p className="text-xs text-textMuted leading-relaxed font-mono bg-panelAlt rounded-lg p-3">
                {report.what_leaves_each_party["Passive Parties"]}
              </p>
            </div>
          </div>

          <div className="panel p-6 mt-8 flex items-start gap-3 border-line">
            <Lock className="w-5 h-5 text-textMuted shrink-0 mt-0.5" />
            <div className="text-sm text-textMuted leading-relaxed">
              <span className="text-textPrimary font-medium">Where privacy stops: </span>
              Once representation synthesis (ICGM) is trained, everything runs locally on the active
              party — no further cross-party communication is needed for training or inference. That's
              the paper's key efficiency argument: privacy protection is concentrated in the four PCA-ROM
              rounds, not repeated on every training step like vanilla VFL.
            </div>
          </div>
        </>
      )}
    </div>
  );
}
