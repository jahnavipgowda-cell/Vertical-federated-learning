import { NavLink } from "react-router-dom";
import { LayoutDashboard, Users, Shuffle, Layers, Sparkles, BarChart3, ShieldCheck, Radio } from "lucide-react";
import { useSession } from "../SessionContext";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, step: null },
  { to: "/setup", label: "Party Setup", icon: Users, step: 1 },
  { to: "/alignment", label: "Data Alignment (PSI)", icon: Shuffle, step: 2 },
  { to: "/extraction", label: "Federated Extraction", icon: Layers, step: 3 },
  { to: "/synthesis", label: "Representation Synthesis", icon: Sparkles, step: 4 },
  { to: "/evaluation", label: "Model Evaluation", icon: BarChart3, step: 5 },
  { to: "/privacy", label: "Privacy & Security", icon: ShieldCheck, step: null },
];

function stepDone(status, step) {
  if (!status?.steps_completed) return false;
  if (step === 1) return status.steps_completed.session;
  if (step === 3) return status.steps_completed.pca_rom;
  if (step === 4) return status.steps_completed.icgm;
  if (step === 5) return status.steps_completed.evaluation;
  if (step === 2) return status.steps_completed.session;
  return false;
}

export default function Layout({ children }) {
  const { status } = useSession();

  return (
    <div className="min-h-screen flex bg-ink">
      <aside className="w-72 shrink-0 border-r border-line bg-panel/60 flex flex-col">
        <div className="px-6 py-6 border-b border-line">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-signal/15 border border-signal/40 flex items-center justify-center">
              <Radio className="w-4 h-4 text-signal" />
            </div>
            <div>
              <div className="font-display font-semibold text-textPrimary leading-tight">VFedRS Studio</div>
              <div className="text-[11px] text-textMuted font-mono">vertical federated learning</div>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV.map((item) => {
            const Icon = item.icon;
            const done = item.step ? stepDone(status, item.step) : false;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition group ${
                    isActive
                      ? "bg-signal/10 text-signal border border-signal/30"
                      : "text-textMuted hover:text-textPrimary hover:bg-panelAlt border border-transparent"
                  }`
                }
              >
                <span className="relative">
                  <Icon className="w-4 h-4" />
                  {done && (
                    <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-signal" />
                  )}
                </span>
                <span className="flex-1">{item.label}</span>
                {item.step && <span className="text-[10px] font-mono text-textMuted/60">0{item.step}</span>}
              </NavLink>
            );
          })}
        </nav>

        <div className="px-4 py-4 border-t border-line">
          {status?.active ? (
            <div className="text-xs space-y-1.5">
              <div className="flex justify-between text-textMuted">
                <span>Dataset</span>
                <span className="text-textPrimary font-mono truncate max-w-[140px]">{status.dataset_label}</span>
              </div>
              <div className="flex justify-between text-textMuted">
                <span>Aligned</span>
                <span className="text-signal font-mono">{status.n_aligned} / {status.n_train}</span>
              </div>
              <div className="flex justify-between text-textMuted">
                <span>Parties</span>
                <span className="text-textPrimary font-mono">{Object.keys(status.party_feature_counts || {}).length}</span>
              </div>
            </div>
          ) : (
            <div className="text-xs text-textMuted">No active session yet — start at Party Setup.</div>
          )}
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto px-8 py-10">{children}</div>
      </main>
    </div>
  );
}
