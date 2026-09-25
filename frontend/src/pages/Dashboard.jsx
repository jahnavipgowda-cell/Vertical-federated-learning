import { Link } from "react-router-dom";
import { ArrowRight, Lock, Network, Sparkles } from "lucide-react";
import NetworkDiagram from "../components/NetworkDiagram";
import StatCard from "../components/StatCard";
import { useSession } from "../SessionContext";

const PIPELINE = [
  { n: "01", title: "Party Setup", desc: "Choose a dataset and split features vertically across an active party and K passive parties.", to: "/setup" },
  { n: "02", title: "Data Alignment", desc: "Simulate private set intersection (PSI) — only a fraction of samples align across all parties.", to: "/alignment" },
  { n: "03", title: "Federated Extraction", desc: "Run PCA-ROM: a masked, removable-orthogonal-mask SVD that extracts shared representations without exposing raw features.", to: "/extraction" },
  { n: "04", title: "Representation Synthesis", desc: "Train the ICGM generator/discriminator so the active party can synthesize representations for its non-aligned samples too.", to: "/synthesis" },
  { n: "05", title: "Model Evaluation", desc: "Compare LOCAL, vanilla VFL, PCA-ROM-only, and full VFedRS on held-out accuracy.", to: "/evaluation" },
];

export default function Dashboard() {
  const { status } = useSession();

  return (
    <div>
      <div className="grid grid-cols-1 lg:grid-cols-[1.3fr_1fr] gap-10 items-center mb-16">
        <div>
          <div className="label-eyebrow mb-3">vertical federated learning · representation synthesis</div>
          <h1 className="text-4xl md:text-5xl font-display font-semibold leading-tight text-textPrimary">
            Give the active party's <span className="text-signal">non-aligned</span> data a seat at the table.
          </h1>
          <p className="text-textMuted mt-5 max-w-xl leading-relaxed">
            Most vertical federated learning only trains on the sliver of samples every party shares.
            VFedRS Studio walks through a working implementation of{" "}
            <span className="text-textPrimary">Vertical Federated Representation Synthesis</span>: a
            privacy-preserving PCA under removable masks, plus a conditional generator that lets the
            active party train on <em>all</em> of its local data — aligned or not — without any party
            ever seeing another's raw features.
          </p>
          <div className="mt-7 flex gap-3">
            <Link to="/setup" className="btn-primary inline-flex items-center gap-2">
              Start a session <ArrowRight className="w-4 h-4" />
            </Link>
            <Link to="/privacy" className="btn-secondary inline-flex items-center gap-2">
              <Lock className="w-4 h-4" /> How privacy is preserved
            </Link>
          </div>
        </div>
        <div className="panel p-6 flex items-center justify-center aspect-square max-w-sm mx-auto w-full">
          <NetworkDiagram nPassive={3} pulse={status?.active} />
        </div>
      </div>

      {status?.active && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
          <StatCard label="Aligned samples" value={status.n_aligned} sub={`of ${status.n_train} training samples`} />
          <StatCard label="Non-aligned" value={status.n_nonaligned} sub="active party only" accent="mask" />
          <StatCard label="Parties" value={Object.keys(status.party_feature_counts || {}).length} sub="1 active + passive" />
          <StatCard label="Classes" value={status.n_classes} sub={status.dataset_label} />
        </div>
      )}

      <div className="mb-6 flex items-center gap-2">
        <Network className="w-4 h-4 text-signal" />
        <h2 className="font-display font-semibold text-lg">The pipeline</h2>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {PIPELINE.map((s) => (
          <Link key={s.n} to={s.to} className="panel p-5 hover:border-signal/50 transition group">
            <div className="flex items-start gap-4">
              <div className="font-mono text-signal/70 text-sm pt-0.5">{s.n}</div>
              <div>
                <div className="font-display font-medium text-textPrimary group-hover:text-signal transition">{s.title}</div>
                <div className="text-sm text-textMuted mt-1 leading-relaxed">{s.desc}</div>
              </div>
            </div>
          </Link>
        ))}
        <div className="panel p-5 border-mask/30 bg-mask/5">
          <div className="flex items-start gap-4">
            <Sparkles className="w-4 h-4 text-mask mt-1 shrink-0" />
            <div>
              <div className="font-display font-medium text-mask">Why it matters</div>
              <div className="text-sm text-textMuted mt-1 leading-relaxed">
                Communication only happens across four fixed rounds in Phase 1 — no iterative gradient
                exchange like vanilla VFL. See the cost comparison on the Evaluation page.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
