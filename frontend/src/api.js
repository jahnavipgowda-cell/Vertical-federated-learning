const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function req(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (e) {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  presets: () => req("/api/presets"),
  sessionStatus: () => req("/api/session/status"),
  newSession: (payload) => req("/api/session/new", { method: "POST", body: JSON.stringify(payload) }),
  psiAlign: (payload) => req("/api/psi/align", { method: "POST", body: JSON.stringify(payload) }),
  psiPreview: () => req("/api/psi/preview"),
  runPcaRom: (payload) => req("/api/pca-rom/run", { method: "POST", body: JSON.stringify(payload) }),
  trainIcgm: (payload) => req("/api/icgm/train", { method: "POST", body: JSON.stringify(payload) }),
  runEvaluation: () => req("/api/evaluate/run", { method: "POST" }),
  privacyReport: () => req("/api/privacy/report"),
  health: () => req("/api/health"),
};
