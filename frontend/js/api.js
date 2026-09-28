/* ==========================================================================
   ChestVision AI — API client
   Thin fetch() wrapper: attaches JWT, parses errors, handles auth redirects.
   ========================================================================== */

const API_BASE = "http://localhost:8000";

const Auth = {
  getToken() { return localStorage.getItem("cv_token"); },
  getRole() { return localStorage.getItem("cv_role"); },
  getName() { return localStorage.getItem("cv_name"); },
  setSession(token, role, name) {
    localStorage.setItem("cv_token", token);
    localStorage.setItem("cv_role", role);
    localStorage.setItem("cv_name", name);
  },
  clearSession() {
    localStorage.removeItem("cv_token");
    localStorage.removeItem("cv_role");
    localStorage.removeItem("cv_name");
  },
  isLoggedIn() { return !!this.getToken(); },
  logout() {
    this.clearSession();
    window.location.href = "/index.html";
  },
  /**
   * Call at the top of every protected page. Redirects to the correct
   * login page if not authenticated, or if logged in as the wrong role.
   */
  requireRole(role) {
    const token = this.getToken();
    const currentRole = this.getRole();
    if (!token || currentRole !== role) {
      window.location.href = role === "admin" ? "/admin-login.html" : "/doctor-login.html";
    }
  },
};

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function apiRequest(path, { method = "GET", body = null, isFormData = false } = {}) {
  const headers = {};
  const token = Auth.getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (!isFormData && body) headers["Content-Type"] = "application/json";

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: isFormData ? body : body ? JSON.stringify(body) : undefined,
    });
  } catch (networkErr) {
    throw new ApiError(
      "Could not reach the ChestVision AI server. Confirm the backend is running.",
      0
    );
  }

  if (response.status === 401) {
    Auth.clearSession();
    const role = Auth.getRole();
    window.location.href = window.location.pathname.startsWith("/doctor")
      ? "/doctor-login.html"
      : "/admin-login.html";
    throw new ApiError("Session expired. Please log in again.", 401);
  }

  let data = null;
  const text = await response.text();
  if (text) {
    try { data = JSON.parse(text); } catch { data = null; }
  }

  if (!response.ok) {
    const detail = (data && (data.detail || data.message)) || `Request failed (${response.status})`;
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
      : detail;
    throw new ApiError(message, response.status);
  }

  return data;
}

const Api = {
  // ---- Auth ----
  registerAdmin: (payload) => apiRequest("/api/auth/admin/register", { method: "POST", body: payload }),
  loginAdmin: (payload) => apiRequest("/api/auth/admin/login", { method: "POST", body: payload }),
  registerDoctor: (payload) => apiRequest("/api/auth/doctor/register", { method: "POST", body: payload }),
  loginDoctor: (payload) => apiRequest("/api/auth/doctor/login", { method: "POST", body: payload }),

  // ---- Admin: dashboard / patients ----
  adminDashboard: () => apiRequest("/api/admin/dashboard"),
  listPatients: (search = "") => apiRequest(`/api/admin/patients${search ? `?search=${encodeURIComponent(search)}` : ""}`),
  getPatient: (id) => apiRequest(`/api/admin/patients/${id}`),
  addPatient: (payload) => apiRequest("/api/admin/patients", { method: "POST", body: payload }),
  updatePatient: (id, payload) => apiRequest(`/api/admin/patients/${id}`, { method: "PUT", body: payload }),
  deletePatient: (id) => apiRequest(`/api/admin/patients/${id}`, { method: "DELETE" }),
  listPatientXrays: (id) => apiRequest(`/api/admin/patients/${id}/xrays`),

  // ---- Admin: x-ray / analysis / reports ----
  uploadXray: (patientId, file) => {
    const form = new FormData();
    form.append("file", file);
    return apiRequest(`/api/admin/patients/${patientId}/xray`, { method: "POST", body: form, isFormData: true });
  },
  startAnalysis: (imageId) => apiRequest(`/api/admin/xray/${imageId}/analyze`, { method: "POST" }),
  analysisHistory: () => apiRequest("/api/admin/analysis-history"),
  generateReport: (resultId) => apiRequest(`/api/admin/analysis/${resultId}/report`, { method: "POST" }),
  listReports: () => apiRequest("/api/admin/reports"),
  getReport: (reportId) => apiRequest(`/api/admin/reports/${reportId}`),

  // ---- Doctor (read-only) ----
  doctorDashboard: () => apiRequest("/api/doctor/dashboard"),
  doctorListReports: () => apiRequest("/api/doctor/reports"),
  doctorGetReport: (reportId) => apiRequest(`/api/doctor/reports/${reportId}`),
  doctorGetPatient: (id) => apiRequest(`/api/doctor/patients/${id}`),
};

/** Resolves a backend-relative filesystem path (e.g. .../uploads/x.png) to a servable URL. */
function fileUrl(absolutePath) {
  if (!absolutePath) return "";
  const normalized = absolutePath.replace(/\\/g, '/');
  const match = normalized.match(/(\/uploads\/|\/reports\/)/i);
  if (match) {
    const relative = normalized.slice(match.index);
    return `${API_BASE}/files${relative}`;
  }
  // Fallback if neither folder name is found
  const parts = normalized.split('/');
  return `${API_BASE}/files/${parts[parts.length - 1]}`;
}
