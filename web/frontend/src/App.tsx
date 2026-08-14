import { Route, Routes } from "react-router-dom";
import { AuthProvider, RequireAuth } from "./auth";
import { Layout } from "./components/Layout";
import { PublicLayout } from "./components/PublicLayout";
import { LiveTrading } from "./pages/LiveTrading";
import { Login } from "./pages/Login";
import { NewRun } from "./pages/NewRun";
import { PublicRunDetail } from "./pages/PublicRunDetail";
import { PublicRunHistory } from "./pages/PublicRunHistory";
import { RunDetail } from "./pages/RunDetail";
import { RunHistory } from "./pages/RunHistory";
import { Strategies } from "./pages/Strategies";

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route element={<PublicLayout />}>
          <Route path="/public" element={<PublicRunHistory />} />
          <Route path="/public/runs/:id" element={<PublicRunDetail />} />
        </Route>
        <Route
          element={
            <RequireAuth>
              <Layout />
            </RequireAuth>
          }
        >
          <Route path="/" element={<RunHistory />} />
          <Route path="/runs/new" element={<NewRun />} />
          <Route path="/runs/:id" element={<RunDetail />} />
          <Route path="/live" element={<Strategies />} />
          <Route path="/live/:strategyId" element={<LiveTrading />} />
        </Route>
      </Routes>
    </AuthProvider>
  );
}
