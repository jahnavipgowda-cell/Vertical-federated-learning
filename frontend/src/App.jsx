import { BrowserRouter, Routes, Route } from "react-router-dom";
import { SessionProvider } from "./SessionContext";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import PartySetup from "./pages/PartySetup";
import DataAlignment from "./pages/DataAlignment";
import RepresentationExtraction from "./pages/RepresentationExtraction";
import RepresentationSynthesis from "./pages/RepresentationSynthesis";
import ModelEvaluation from "./pages/ModelEvaluation";
import PrivacySecurity from "./pages/PrivacySecurity";

export default function App() {
  return (
    <SessionProvider>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/setup" element={<PartySetup />} />
            <Route path="/alignment" element={<DataAlignment />} />
            <Route path="/extraction" element={<RepresentationExtraction />} />
            <Route path="/synthesis" element={<RepresentationSynthesis />} />
            <Route path="/evaluation" element={<ModelEvaluation />} />
            <Route path="/privacy" element={<PrivacySecurity />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </SessionProvider>
  );
}
