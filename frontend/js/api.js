/* ==========================================================================
   ChestVision AI — API client
   Thin fetch() wrapper: attaches JWT, parses errors, handles auth redirects.
   ========================================================================== */

const API_BASE = "http://localhost:8000";


/* ==========================================================================
   Authentication / Session
   ========================================================================== */

const Auth = {

  // ------------------------------------------------------------
  // Get stored session information
  // ------------------------------------------------------------

  getToken() {
    return localStorage.getItem("cv_token");
  },

  getRole() {
    return localStorage.getItem("cv_role");
  },

  getName() {
    return localStorage.getItem("cv_name");
  },

  getEmail() {
    return localStorage.getItem("cv_email");
  },


  // ------------------------------------------------------------
  // Save session information
  // ------------------------------------------------------------

  setSession(token, role, name, email) {

    localStorage.setItem("cv_token", token);
    localStorage.setItem("cv_role", role);

    // Store actual name
    localStorage.setItem("cv_name", name || "");

    // Store actual email
    localStorage.setItem("cv_email", email || "");
  },


  // ------------------------------------------------------------
  // Clear session
  // ------------------------------------------------------------

  clearSession() {

    localStorage.removeItem("cv_token");
    localStorage.removeItem("cv_role");
    localStorage.removeItem("cv_name");
    localStorage.removeItem("cv_email");
  },


  // ------------------------------------------------------------
  // Check login
  // ------------------------------------------------------------

  isLoggedIn() {
    return !!this.getToken();
  },


  // ------------------------------------------------------------
  // Logout
  // ------------------------------------------------------------

  logout() {

    this.clearSession();

    window.location.href = "/index.html";
  },


  // ------------------------------------------------------------
  // Protect pages based on role
  // ------------------------------------------------------------

  requireRole(role) {

    const token = this.getToken();
    const currentRole = this.getRole();

    if (!token || currentRole !== role) {

      if (role === "admin") {
        window.location.href = "/admin-login.html";
      } else {
        window.location.href = "/doctor-login.html";
      }

      return false;
    }

    return true;
  }
};


/* ==========================================================================
   API Error
   ========================================================================== */

class ApiError extends Error {

  constructor(message, status) {

    super(message);

    this.name = "ApiError";
    this.status = status;
  }
}


/* ==========================================================================
   Generic API Request
   ========================================================================== */

async function apiRequest(
  path,
  {
    method = "GET",
    body = null,
    isFormData = false
  } = {}
) {

  const headers = {};

  const token = Auth.getToken();

  // Attach JWT
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  // JSON body
  if (!isFormData && body) {
    headers["Content-Type"] = "application/json";
  }

  let response;

  try {

    response = await fetch(`${API_BASE}${path}`, {

      method,

      headers,

      body: isFormData
        ? body
        : body
          ? JSON.stringify(body)
          : undefined
    });

  } catch (networkErr) {

    console.error("Network error:", networkErr);

    throw new ApiError(
      "Could not reach the ChestVision AI server. Confirm that the backend is running.",
      0
    );
  }


  /* ------------------------------------------------------------
     Handle unauthorized
     ------------------------------------------------------------ */

  if (response.status === 401) {

    const currentRole = Auth.getRole();

    Auth.clearSession();

    if (currentRole === "doctor") {

      window.location.href = "/doctor-login.html";

    } else {

      window.location.href = "/admin-login.html";
    }

    throw new ApiError(
      "Session expired. Please log in again.",
      401
    );
  }


  /* ------------------------------------------------------------
     Read response
     ------------------------------------------------------------ */

  let data = null;

  const text = await response.text();

  if (text) {

    try {

      data = JSON.parse(text);

    } catch {

      data = null;
    }
  }


  /* ------------------------------------------------------------
     Handle errors
     ------------------------------------------------------------ */

  if (!response.ok) {

    const detail =
      data &&
      (
        data.detail ||
        data.message
      );

    const message = Array.isArray(detail)
      ? detail
          .map((d) => d.msg || JSON.stringify(d))
          .join("; ")
      : detail || `Request failed (${response.status})`;

    throw new ApiError(
      message,
      response.status
    );
  }


  return data;
}


/* ==========================================================================
   API
   ========================================================================== */

const Api = {

  /* ============================================================
     Authentication
     ============================================================ */

  registerAdmin: (payload) =>
    apiRequest(
      "/api/auth/admin/register",
      {
        method: "POST",
        body: payload
      }
    ),

  loginAdmin: (payload) =>
    apiRequest(
      "/api/auth/admin/login",
      {
        method: "POST",
        body: payload
      }
    ),

  registerDoctor: (payload) =>
    apiRequest(
      "/api/auth/doctor/register",
      {
        method: "POST",
        body: payload
      }
    ),

  loginDoctor: (payload) =>
    apiRequest(
      "/api/auth/doctor/login",
      {
        method: "POST",
        body: payload
      }
    ),


  /* ============================================================
     Admin — Dashboard / Patients
     ============================================================ */

  adminDashboard: () =>
    apiRequest(
      "/api/admin/dashboard"
    ),

  listPatients: (search = "") =>
    apiRequest(
      `/api/admin/patients${
        search
          ? `?search=${encodeURIComponent(search)}`
          : ""
      }`
    ),

  getPatient: (id) =>
    apiRequest(
      `/api/admin/patients/${id}`
    ),

  addPatient: (payload) =>
    apiRequest(
      "/api/admin/patients",
      {
        method: "POST",
        body: payload
      }
    ),

  updatePatient: (id, payload) =>
    apiRequest(
      `/api/admin/patients/${id}`,
      {
        method: "PUT",
        body: payload
      }
    ),

  deletePatient: (id) =>
    apiRequest(
      `/api/admin/patients/${id}`,
      {
        method: "DELETE"
      }
    ),

  listPatientXrays: (id) =>
    apiRequest(
      `/api/admin/patients/${id}/xrays`
    ),


  /* ============================================================
     Admin — X-Ray / Analysis / Reports
     ============================================================ */

  uploadXray: (patientId, file) => {

    const form = new FormData();

    form.append("file", file);

    return apiRequest(
      `/api/admin/patients/${patientId}/xray`,
      {
        method: "POST",
        body: form,
        isFormData: true
      }
    );
  },

  startAnalysis: (imageId) =>
    apiRequest(
      `/api/admin/xray/${imageId}/analyze`,
      {
        method: "POST"
      }
    ),

  analysisHistory: () =>
    apiRequest(
      "/api/admin/analysis-history"
    ),

  generateReport: (resultId) =>
    apiRequest(
      `/api/admin/analysis/${resultId}/report`,
      {
        method: "POST"
      }
    ),

  listReports: () =>
    apiRequest(
      "/api/admin/reports"
    ),

  getReport: (reportId) =>
    apiRequest(
      `/api/admin/reports/${reportId}`
    ),


  /* ============================================================
     Doctor — Dashboard / Reports
     ============================================================ */

  doctorDashboard: () =>
    apiRequest(
      "/api/doctor/dashboard"
    ),

  doctorListReports: () =>
    apiRequest(
      "/api/doctor/reports"
    ),

  doctorGetReport: (reportId) =>
    apiRequest(
      `/api/doctor/reports/${reportId}`
    ),

  doctorGetPatient: (id) =>
    apiRequest(
      `/api/doctor/patients/${id}`
    ),

  doctorUpdateReportStatus: (reportId, payload) =>
    apiRequest(
      `/api/doctor/reports/${reportId}/status`,
      {
        method: "PATCH",
        body: payload
      }
    )
};


/* ==========================================================================
   File URL Helper
   ========================================================================== */

function fileUrl(absolutePath) {

  if (!absolutePath) {
    return "";
  }

  const normalized =
    absolutePath.replace(/\\/g, "/");

  const match =
    normalized.match(/(\/uploads\/|\/reports\/)/i);

  if (match) {

    const relative =
      normalized.slice(match.index);

    return `${API_BASE}/files${relative}`;
  }

  const parts =
    normalized.split("/");

  return `${API_BASE}/files/${parts[parts.length - 1]}`;
}