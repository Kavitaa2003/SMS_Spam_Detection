// Small shared helper: keeps the JWT in localStorage and attaches it as a
// Bearer token to authenticated requests. This is the "fullstack talking to
// an API with JWT" pattern — same idea a React app would use, just without
// a build step.

const TOKEN_KEY = 'spam_detector_token';
const USERNAME_KEY = 'spam_detector_username';

const Auth = {
  getToken() { return localStorage.getItem(TOKEN_KEY); },
  getUsername() { return localStorage.getItem(USERNAME_KEY); },
  isLoggedIn() { return !!this.getToken(); },
  setSession(token, username) {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USERNAME_KEY, username);
  },
  clearSession() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USERNAME_KEY);
  },
};

async function apiFetch(path, options = {}) {
  const headers = options.headers || {};
  headers['Content-Type'] = 'application/json';
  if (Auth.isLoggedIn()) {
    headers['Authorization'] = `Bearer ${Auth.getToken()}`;
  }
  const res = await fetch(path, { ...options, headers });
  let data = null;
  try { data = await res.json(); } catch (_) { /* no body */ }
  if (!res.ok) {
    const message = (data && data.detail) || `Request failed (${res.status})`;
    throw new Error(message);
  }
  return data;
}

function renderNav() {
  const linksEl = document.getElementById('nav-links');
  if (!linksEl) return;

  if (Auth.isLoggedIn()) {
    linksEl.innerHTML = `
      <a href="/">Check</a>
      <a href="/history">History</a>
      <span class="mono" style="color: var(--text-dim); font-size: 13px;">${Auth.getUsername()}</span>
      <button id="logout-btn">Log out</button>
    `;
    document.getElementById('logout-btn').addEventListener('click', () => {
      Auth.clearSession();
      window.location.href = '/';
    });
  } else {
    linksEl.innerHTML = `
      <a href="/">Check</a>
      <a href="/login">Log in</a>
      <a href="/register">Sign up</a>
    `;
  }
}

document.addEventListener('DOMContentLoaded', renderNav);
