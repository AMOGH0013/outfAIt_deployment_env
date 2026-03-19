const AUTH_TOKEN_KEY = "wai_token";
const AUTH_COOKIE_NAME = "wai_token";
const AUTH_COOKIE_MAX_AGE_SECONDS = 60 * 60;
const THEME_KEY = "wai_theme";

function readCookie(name) {
  const prefix = `${name}=`;
  const raw = document.cookie
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith(prefix));
  return raw ? decodeURIComponent(raw.slice(prefix.length)) : "";
}

function writeTokenCookie(token, maxAgeSeconds = AUTH_COOKIE_MAX_AGE_SECONDS) {
  if (!token) return;
  document.cookie = `${AUTH_COOKIE_NAME}=${encodeURIComponent(token)}; Path=/; SameSite=Lax; Max-Age=${maxAgeSeconds}`;
}

function clearTokenCookie() {
  document.cookie = `${AUTH_COOKIE_NAME}=; Path=/; SameSite=Lax; Max-Age=0`;
}

function setToken(token) {
  if (!token) return;
  localStorage.setItem(AUTH_TOKEN_KEY, token);
  writeTokenCookie(token);
}

function clearToken() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  clearTokenCookie();
}

function getToken() {
  const localToken = localStorage.getItem(AUTH_TOKEN_KEY) || "";
  const cookieToken = readCookie(AUTH_COOKIE_NAME);
  const token = localToken || cookieToken;
  if (!token) return null;

  if (!localToken) localStorage.setItem(AUTH_TOKEN_KEY, token);
  if (!cookieToken) writeTokenCookie(token);
  return token;
}

function redirectToLogin() {
  const next = encodeURIComponent(location.pathname + location.search);
  location.href = `/login?next=${next}`;
}

function logout() {
  clearToken();
  redirectToLogin();
}

function requireAuth() {
  if (!getToken()) {
    redirectToLogin();
    return false;
  }
  return true;
}

function isAuthEndpoint(url) {
  const value = String(url || "");
  return value.includes("/auth/token");
}

async function apiFetch(url, options = {}) {
  const token = getToken();
  if (!token && !isAuthEndpoint(url)) {
    redirectToLogin();
    throw new Error("Authentication required. Redirecting to login.");
  }

  const headers = new Headers(options.headers || {});
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const isFormData = options.body instanceof FormData;
  if (!isFormData && options.body && typeof options.body === "string" && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(url, {
    cache: "no-store",
    credentials: "same-origin",
    ...options,
    headers,
  });

  if (res.status === 401 && !isAuthEndpoint(url)) {
    clearToken();
    redirectToLogin();
    throw new Error("Session expired. Redirecting to login.");
  }

  const contentType = res.headers.get("content-type") || "";
  let payload = null;
  if (contentType.includes("application/json")) {
    payload = await res.json().catch(() => null);
  } else {
    payload = await res.text().catch(() => null);
  }

  if (!res.ok) {
    const detail = payload && typeof payload === "object" && payload.detail ? payload.detail : payload;
    throw new Error(`${res.status} ${res.statusText}: ${detail || "Request failed"}`);
  }
  return payload;
}

function renderImage(url) {
  return url || "";
}

function fmtDateISO(value) {
  const dt = new Date(value);
  const yyyy = dt.getFullYear();
  const mm = String(dt.getMonth() + 1).padStart(2, "0");
  const dd = String(dt.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function fmtMonthISO(value) {
  const dt = new Date(value);
  const yyyy = dt.getFullYear();
  const mm = String(dt.getMonth() + 1).padStart(2, "0");
  return `${yyyy}-${mm}`;
}

function showToast(message, type = "info") {
  if (!message) return;
  let host = document.getElementById("toastHost");
  if (!host) {
    host = document.createElement("div");
    host.id = "toastHost";
    host.className = "toast-host";
    document.body.appendChild(host);
  }
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  host.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 220);
  }, 3000);
}

const pushToast = showToast;

function loadTheme() {
  return localStorage.getItem(THEME_KEY) || "dark";
}

function applyTheme(theme) {
  const normalized = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", normalized);
  localStorage.setItem(THEME_KEY, normalized);
  const toggle = document.getElementById("themeToggle");
  if (toggle) toggle.textContent = normalized === "dark" ? "Light" : "Dark";
}

function toggleTheme() {
  const next = loadTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
}

function getColorHex(name) {
  const value = String(name || "").toLowerCase().trim();
  const table = [
    ["hot pink", "#ff4f9f"],
    ["baby pink", "#f7c9de"],
    ["dusty rose", "#ca8b98"],
    ["blush pink", "#f2b6c4"],
    ["salmon", "#ff9076"],
    ["coral", "#f27e6a"],
    ["pink", "#ef69ae"],
    ["magenta", "#cc278f"],
    ["purple", "#7e58c9"],
    ["royal purple", "#6f43bf"],
    ["lavender", "#bca7ed"],
    ["lilac", "#bca0da"],
    ["indigo", "#4a45a7"],
    ["royal blue", "#3e60de"],
    ["navy", "#20386a"],
    ["sky blue", "#71b7ee"],
    ["powder blue", "#a7caef"],
    ["teal", "#179f98"],
    ["turquoise", "#2ebbb9"],
    ["mint", "#98dfcc"],
    ["forest green", "#2f7d4f"],
    ["olive", "#7a8d35"],
    ["green", "#37a56d"],
    ["khaki", "#a99c62"],
    ["mustard", "#c2a129"],
    ["yellow", "#ddbe44"],
    ["rust", "#b66138"],
    ["burnt orange", "#c95c23"],
    ["orange", "#f2852f"],
    ["red", "#d14e56"],
    ["burgundy", "#7f2f45"],
    ["brown", "#885b39"],
    ["camel", "#ba8f61"],
    ["beige", "#d6c09e"],
    ["taupe", "#9a8776"],
    ["tan", "#b7936e"],
    ["off white", "#f5efe2"],
    ["cream", "#f7eed5"],
    ["white", "#f8f8f6"],
    ["cool gray", "#93a3b4"],
    ["warm gray", "#9e9286"],
    ["light gray", "#bec7d1"],
    ["dark gray", "#656c74"],
    ["gray", "#9098a3"],
    ["grey", "#9098a3"],
    ["charcoal", "#3b414a"],
    ["black", "#12161e"],
  ];
  for (const [key, hex] of table) {
    if (value.includes(key)) return hex;
  }
  return "#8793aa";
}

function getColorFamily(name) {
  const value = String(name || "").toLowerCase();
  if (!value) return "neutral";
  if (
    value.includes("black") ||
    value.includes("charcoal") ||
    value.includes("dark gray") ||
    value.includes("dark grey") ||
    value.includes("navy")
  ) {
    return "dark";
  }
  const warmKeys = [
    "red",
    "orange",
    "rust",
    "burnt",
    "pink",
    "coral",
    "salmon",
    "magenta",
    "burgundy",
    "brown",
    "camel",
    "beige",
    "tan",
    "mustard",
    "yellow",
    "khaki",
  ];
  if (warmKeys.some((k) => value.includes(k))) return "warm";

  const coolKeys = [
    "blue",
    "teal",
    "turquoise",
    "green",
    "mint",
    "olive",
    "purple",
    "indigo",
    "lavender",
    "lilac",
  ];
  if (coolKeys.some((k) => value.includes(k))) return "cool";
  return "neutral";
}

function stableVariantIndex(seedValue, length) {
  const text = String(seedValue || "");
  if (!text || !length) return 0;
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % length;
}

function summarizeExplanation(expl) {
  if (!expl || typeof expl !== "object") {
    return "Strong pick: clean balance, fresh rotation, easy confidence.";
  }

  const reasoning = expl.reasoning && typeof expl.reasoning === "object" ? expl.reasoning : {};
  const topColor = String(expl.top_color || "").trim().toLowerCase();
  const bottomColor = String(expl.bottom_color || "").trim().toLowerCase();
  const score = Number(expl.final_score ?? 0);

  const openerPool = [
    "Confident look",
    "Strong style call",
    "Sharp outfit choice",
    "High-impact pairing",
  ];
  const seed = `${expl.top || ""}-${expl.bottom || ""}-${topColor}-${bottomColor}-${score.toFixed(2)}`;
  const opener = openerPool[stableVariantIndex(seed, openerPool.length)];

  const notes = [];
  if (reasoning.color) {
    if (topColor && bottomColor) {
      notes.push(`${topColor} + ${bottomColor} feels intentional`);
    } else {
      notes.push("color balance looks polished");
    }
  }
  if (reasoning.fit || reasoning.body_shape) notes.push("fit profile stays flattering");
  if (reasoning.style) notes.push("it aligns with your style direction");
  if (reasoning.novelty || reasoning.embedding_diversity) notes.push("it keeps your rotation fresh");
  if (reasoning.feedback || reasoning.collaborative) notes.push("it is backed by your wear pattern");

  if (!notes.length) {
    if (score >= 0.8) return `${opener}: polished, modern, and ready to wear.`;
    if (score >= 0.6) return `${opener}: balanced, reliable, and easy to carry.`;
    return `${opener}: clean structure with room to personalize.`;
  }

  const concise = notes.slice(0, 2).join("; ");
  return `${opener}: ${concise}.`;
}

function simpleReasonText(expl) {
  if (!expl || typeof expl !== "object") {
    return {
      colorReason: "The colors land clean together and give a polished finish.",
      styleReason: "The silhouette pairing feels balanced and confidently styled.",
    };
  }
  const reasoning = expl.reasoning && typeof expl.reasoning === "object" ? expl.reasoning : {};
  return {
    colorReason:
      typeof reasoning.color === "string" && reasoning.color.trim()
        ? reasoning.color.replace(/^good color harmony$/i, "The color pairing is clean, modern, and easy to wear.")
        : "The color pairing is clean, modern, and easy to wear.",
    styleReason:
      typeof reasoning.style === "string" && reasoning.style.trim()
        ? reasoning.style.replace(/^matches your learned style preferences$/i, "This match fits your style profile and looks intentional.")
        : "The shape and formality level of the pieces are a confident match.",
  };
}

function computePersonalizationStatus(profile, hasFeedback) {
  const p = profile && typeof profile === "object" ? profile : {};
  const features = {
    fit_preference: Boolean(p.fit_preference),
    body_shape: Boolean(p.body_shape),
    skin_tone: Boolean(p.skin_tone),
    feedback: Boolean(hasFeedback),
  };
  const score =
    (features.fit_preference ? 1 : 0) +
    (features.body_shape ? 1 : 0) +
    (features.skin_tone ? 1 : 0) +
    (features.feedback ? 1 : 0);

  let level = "Basic";
  if (score === 2) level = "Medium";
  if (score >= 3) level = "High";
  return { level, score, features };
}

function installShortcuts() {
  if (window.__waiShortcutsInstalled) return;
  window.__waiShortcutsInstalled = true;

  document.addEventListener("keydown", (event) => {
    const isTyping =
      event.target instanceof HTMLElement &&
      (event.target.tagName === "INPUT" ||
        event.target.tagName === "TEXTAREA" ||
        event.target.tagName === "SELECT" ||
        event.target.isContentEditable);
    if (isTyping) return;

    const key = event.key.toLowerCase();
    if (key === "s") location.href = "/frontend/scan.html";
    else if (key === "w") location.href = "/frontend/wardrobe.html";
    else if (key === "o") location.href = "/frontend/outfits.html";
    else if (event.key === "?" || (event.shiftKey && event.key === "/")) {
      event.preventDefault();
      toggleShortcutModal(true);
    } else if (event.key === "Escape") {
      toggleShortcutModal(false);
    }
  });
}

function ensureShortcutModal() {
  if (document.getElementById("shortcutModal")) return;
  const modal = document.createElement("div");
  modal.id = "shortcutModal";
  modal.className = "shortcut-modal";
  modal.innerHTML = `
    <div class="shortcut-card">
      <div class="row space-between">
        <strong>Keyboard Shortcuts</strong>
        <button class="icon-btn" id="shortcutClose" type="button">X</button>
      </div>
      <div class="shortcut-row"><span>Go to Scan</span><span class="shortcut-key">S</span></div>
      <div class="shortcut-row"><span>Go to Wardrobe</span><span class="shortcut-key">W</span></div>
      <div class="shortcut-row"><span>Go to Outfits</span><span class="shortcut-key">O</span></div>
      <div class="shortcut-row"><span>Show this help</span><span class="shortcut-key">?</span></div>
    </div>
  `;
  modal.addEventListener("click", (e) => {
    if (e.target === modal) toggleShortcutModal(false);
  });
  document.body.appendChild(modal);
  const closeBtn = document.getElementById("shortcutClose");
  if (closeBtn) closeBtn.addEventListener("click", () => toggleShortcutModal(false));
}

function toggleShortcutModal(forceOpen) {
  const modal = document.getElementById("shortcutModal");
  if (!modal) return;
  const shouldOpen = typeof forceOpen === "boolean" ? forceOpen : !modal.classList.contains("open");
  modal.classList.toggle("open", shouldOpen);
}

function buildNavbar(activePage = "home") {
  const existing = document.getElementById("topNav");
  if (existing) existing.remove();

  const nav = document.createElement("header");
  nav.className = "top-nav";
  nav.id = "topNav";
  nav.innerHTML = `
    <div class="top-nav-inner">
      <a class="brand-pill" href="/">
        <span class="brand-dot"></span>
        Outfit AI
      </a>
      <nav class="nav-links">
        <a class="nav-link" data-nav="home" href="/">Dashboard</a>
        <a class="nav-link" data-nav="scan" href="/frontend/scan.html">Scan</a>
        <a class="nav-link" data-nav="wardrobe" id="wardrobeNavLink" href="/frontend/wardrobe.html">Wardrobe<span id="wardrobeNavDot" class="nav-dot hidden"></span></a>
        <a class="nav-link" data-nav="outfits" href="/frontend/outfits.html">Outfits</a>
        <a class="nav-link" data-nav="calendar" href="/frontend/calendar.html">Calendar</a>
        <a class="nav-link" data-nav="palette" href="/frontend/palette.html">Palette</a>
        <a class="nav-link" data-nav="history" href="/frontend/history.html">History</a>
        <a class="nav-link" data-nav="profile" href="/frontend/profile.html">Profile</a>
      </nav>
      <div class="nav-actions">
        <button class="theme-toggle" id="themeToggle" type="button">Light</button>
        <button class="icon-btn" id="shortcutBtn" type="button">?</button>
        <div class="avatar" id="userAvatar">U</div>
        <button class="icon-btn" id="logoutBtn" type="button">Logout</button>
      </div>
    </div>
  `;
  document.body.insertBefore(nav, document.body.firstChild);

  document.querySelectorAll(".nav-link").forEach((link) => {
    link.classList.toggle("active", link.dataset.nav === activePage);
  });

  const toggle = document.getElementById("themeToggle");
  if (toggle) toggle.addEventListener("click", toggleTheme);
  const logoutBtn = document.getElementById("logoutBtn");
  if (logoutBtn) logoutBtn.addEventListener("click", logout);
  const shortcutBtn = document.getElementById("shortcutBtn");
  if (shortcutBtn) shortcutBtn.addEventListener("click", () => toggleShortcutModal(true));
}

async function hydrateUserAvatar() {
  const avatar = document.getElementById("userAvatar");
  if (!avatar) return;
  avatar.textContent = "U";
  try {
    const profile = await apiFetch("/body-profile", { method: "GET" });
    const rawName = profile && profile.user_name ? String(profile.user_name) : "User";
    const bits = rawName.trim().split(/\s+/).filter(Boolean);
    const initials = bits.length >= 2 ? `${bits[0][0]}${bits[1][0]}` : rawName.slice(0, 2);
    avatar.textContent = initials.toUpperCase();
  } catch (_e) {
    avatar.textContent = "U";
  }
}

async function refreshWardrobeNotificationDot() {
  const dot = document.getElementById("wardrobeNavDot");
  if (!dot) return;
  try {
    const res = await apiFetch("/wardrobe?include_inactive=false");
    const items = Array.isArray(res.items) ? res.items : [];
    const now = Date.now();
    const hasRecent = items.some((item) => {
      const ts = Date.parse(item.created_at || "");
      return Number.isFinite(ts) && now - ts <= 24 * 60 * 60 * 1000;
    });
    dot.classList.toggle("hidden", !hasRecent);
  } catch (_e) {
    dot.classList.add("hidden");
  }
}

function mountAppShell(options = {}) {
  const opts = options || {};
  const requiresAuth = opts.requiresAuth !== false;
  if (requiresAuth && !requireAuth()) return false;

  applyTheme(loadTheme());
  ensureShortcutModal();
  installShortcuts();
  buildNavbar(opts.activePage || "home");
  hydrateUserAvatar();
  refreshWardrobeNotificationDot();
  return true;
}
