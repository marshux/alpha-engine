import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { Navigate } from "react-router-dom";
import { api } from "./api";
import type { AuthMe } from "./types";

interface AuthState {
  loading: boolean;
  me: AuthMe | null;
  refresh: () => void;
}

const AuthContext = createContext<AuthState>({ loading: true, me: null, refresh: () => {} });

export function AuthProvider({ children }: { children: ReactNode }) {
  const [loading, setLoading] = useState(true);
  const [me, setMe] = useState<AuthMe | null>(null);

  const refresh = () => {
    setLoading(true);
    api
      .me()
      .then(setMe)
      .catch(() => setMe({ authenticated: false, username: null }))
      .finally(() => setLoading(false));
  };

  useEffect(refresh, []);

  return <AuthContext.Provider value={{ loading, me, refresh }}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}

export function RequireAuth({ children }: { children: ReactNode }) {
  const { loading, me } = useAuth();
  if (loading) {
    return <div className="p-8 text-slate-400">Loading...</div>;
  }
  if (!me?.authenticated) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
}
