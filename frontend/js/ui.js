/* ==========================================================================
   ChestVision AI — shared UI helpers
   Toasts, sidebar/topbar injection, status badges, confidence gauges.
   ========================================================================== */

function showToast(message, type = "default") {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    document.body.appendChild(container);
  }
  const toast = document.createElement("div");
  toast.className = `toast ${type === "error" ? "toast-error" : type === "success" ? "toast-success" : ""}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}

function apiErrorToast(err, fallback = "Something went wrong.") {
  showToast(err && err.message ? err.message : fallback, "error");
}

const ICONS = {
  logo: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 6 5 6 9c0 3 1 4 1 6 0 3-2 4-2 6h14c0-2-2-3-2-6 0-2 1-3 1-6 0-4-2-7-6-7z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M9 9c0 2 .8 3 1.5 4M15 9c0 2-.8 3-1.5 4" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>`,
  dashboard: `<svg width="17" height="17" viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="8" height="8" rx="1.5" stroke="currentColor" stroke-width="1.8"/><rect x="13" y="3" width="8" height="5" rx="1.5" stroke="currentColor" stroke-width="1.8"/><rect x="13" y="10" width="8" height="11" rx="1.5" stroke="currentColor" stroke-width="1.8"/><rect x="3" y="13" width="8" height="8" rx="1.5" stroke="currentColor" stroke-width="1.8"/></svg>`,
  patients: `<svg width="17" height="17" viewBox="0 0 24 24" fill="none"><circle cx="9" cy="8" r="3.2" stroke="currentColor" stroke-width="1.8"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="17.5" cy="9" r="2.3" stroke="currentColor" stroke-width="1.6"/><path d="M15 20c.2-2.4 1.7-4.3 3.7-4.9" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
  addPatient: `<svg width="17" height="17" viewBox="0 0 24 24" fill="none"><circle cx="10" cy="8" r="3.2" stroke="currentColor" stroke-width="1.8"/><path d="M3 20c0-3.3 3-6 7-6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M17 13v6M14 16h6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`,
  history: `<svg width="17" height="17" viewBox="0 0 24 24" fill="none"><path d="M4 12a8 8 0 1 1 2.6 5.9" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M4 8v4h4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 8v4l3 2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`,
  reports: `<svg width="17" height="17" viewBox="0 0 24 24" fill="none"><path d="M7 3h7l4 4v14H7z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M14 3v4h4" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M9.5 13h5M9.5 16h5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
  profile: `<svg width="17" height="17" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="3.6" stroke="currentColor" stroke-width="1.8"/><path d="M4.5 20c1-4 4-6 7.5-6s6.5 2 7.5 6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`,
  upload: `<svg width="17" height="17" viewBox="0 0 24 24" fill="none"><path d="M12 15V4M8 8l4-4 4 4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`,
};

const ADMIN_NAV = [
  { href: "dashboard.html", label: "Dashboard", icon: "dashboard" },
  { href: "patients.html", label: "Patients", icon: "patients" },
  { href: "add-patient.html", label: "Add Patient", icon: "addPatient" },
  { href: "reports.html", label: "Reports", icon: "reports" },
  { href: "profile.html", label: "Profile", icon: "profile" },
];

const DOCTOR_NAV = [
  { href: "dashboard.html", label: "Dashboard", icon: "dashboard" },
  { href: "profile.html", label: "Profile", icon: "profile" },
];

/**
 * Renders the sidebar + topbar shell into #app-shell-root, then moves any
 * existing page content into the .content slot. Call once per page after
 * Auth.requireRole() has passed.
 */
function renderShell({ role, pageTitle, pageSub = "" }) {
  const nav = role === "admin" ? ADMIN_NAV : DOCTOR_NAV;
  const currentFile = window.location.pathname.split("/").pop();

  const navHtml = nav
    .map(
      (item) => `<a href="${item.href}" class="${item.href === currentFile ? "active" : ""}">
        ${ICONS[item.icon]}<span class="nav-text">${item.label}</span>
      </a>`
    )
    .join("");

  const root = document.getElementById("app-shell-root");
  const existingContent = root.innerHTML;

  root.innerHTML = `
    <div class="app-shell">
      <aside class="sidebar">
        <div class="sidebar-brand">
          <div class="brand-mark">${ICONS.logo} ChestVision AI</div>
          <div class="brand-sub">${role === "admin" ? "Admin Console" : "Doctor Portal"}</div>
        </div>
        <nav class="sidebar-nav">${navHtml}</nav>
        <div class="sidebar-footer">
          <button id="logout-btn" type="button">Logout</button>
        </div>
      </aside>
      <div class="main">
        <header class="topbar">
          <div>
            <h1>${pageTitle}</h1>
            ${pageSub ? `<div class="topbar-sub">${pageSub}</div>` : ""}
          </div>
          <div class="topbar-user">
            <span class="role-pill">${role}</span>
            <span>${Auth.getName() || ""}</span>
          </div>
        </header>
        <main class="content" id="page-content">${existingContent}</main>
      </div>
    </div>
  `;

  document.getElementById("logout-btn").addEventListener("click", () => Auth.logout());
}

function statusBadgeHtml(priority) {
  const map = {
    Normal: "badge-normal",
    High: "badge-high",
    Urgent: "badge-urgent",
  };
  const cls = map[priority] || "badge-normal";
  return `<span class="badge ${cls}"><span class="badge-dot"></span>${priority}</span>`;
}

function uncertaintyBadgeHtml(status) {
  const map = {
    "High Confidence": "badge-normal",
    "Moderate Confidence": "badge-high",
    "Requires Review": "badge-review",
  };
  const cls = map[status] || "badge-review";
  return `<span class="badge ${cls}"><span class="badge-dot"></span>${status}</span>`;
}

function demoBadgeHtml(isDemo) {
  return isDemo ? `<span class="badge badge-demo">DEMO MODE</span>` : "";
}

/** Renders the signature "vitals gauge" rows for a list of {label, confidence}. */
function gaugeListHtml(conditions, limit = 6) {
  return conditions
    .slice(0, limit)
    .map((c) => {
      const pct = Math.round(c.confidence * 100);
      let color = "var(--status-normal)";
      if (pct >= 75) color = "var(--status-urgent)";
      else if (pct >= 50) color = "var(--status-high)";
      return `
        <div class="gauge-row">
          <div class="gauge-label">${c.label}</div>
          <div class="gauge-track"><div class="gauge-fill" style="width:${pct}%; background:${color};"></div></div>
          <div class="gauge-value">${pct}%</div>
        </div>`;
    })
    .join("");
}

function formatDate(isoString) {
  if (!isoString) return "—";
  
  // Extract strictly the date component (YYYY-MM-DD) to completely ignore time and timezone
  const datePart = isoString.split('T')[0].split(' ')[0];
  
  // Parse the date. JS parses YYYY-MM-DD as UTC midnight.
  const d = new Date(datePart);
  if (isNaN(d.getTime())) return "Date unavailable";
  
  try {
    // Formatting as DD Mon YYYY, enforcing UTC so the date doesn't shift locally
    return d.toLocaleDateString("en-GB", {
      day: "numeric",
      month: "short",
      year: "numeric",
      timeZone: "UTC"
    });
  } catch (e) {
    return "Date unavailable";
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}
