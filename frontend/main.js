/* ════════════════════════════════════════════════════════════════════════
   HIEROGLYPH PIPELINE — frontend logic
   Image  →  POST /api/analyze  →  image + EN + AR + sentiment + intention
   ════════════════════════════════════════════════════════════════════════ */

/* 👇 الـ URL الافتراضي = نفس الدومين الثابت اللي في النوتبوك.
   لو ngrok طلّع URL مختلف، غيّره هنا أو من خانة "Backend" فوق في الموقع. */
const DEFAULT_API_BASE = 'https://irretrievably-unsimpering-darrin.ngrok-free.dev';

const NGROK_HEADERS = { 'ngrok-skip-browser-warning': 'true' };
const RITUAL_STAGES = 6;
const STAGE_MS      = 850;   // وقت كل مرحلة في الـ animation
const MIN_RITUAL_MS = 2200;  // أقل وقت يظهر فيه الـ loading عشان الإحساس يبقى حلو

/* ── elements ─────────────────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const fileInput   = $('fileInput');
const dropZone    = $('dropZone');
const dropPrompt  = $('dropPrompt');
const previewWrap = $('previewWrap');
const dropPreview = $('dropPreview');
const analyzeBtn  = $('analyzeBtn');
const confSlider  = $('confSlider');
const confVal     = $('confVal');
const ritual      = $('ritual');
const results     = $('results');
const errorBox    = $('errorBox');
const errorMsg    = $('errorMsg');
const backendUrl  = $('backendUrl');
const backendDot  = $('backendDot');

let selectedFile = null;
let ritualTimer  = null;

/* ── backend URL (persisted) ──────────────────────────────────────────── */
function getApiBase() {
    return (backendUrl.value || DEFAULT_API_BASE).trim().replace(/\/+$/, '');
}
function loadSavedBackend() {
    let saved = '';
    try { saved = localStorage.getItem('hp_backend') || ''; } catch (_) {}
    backendUrl.value = saved || DEFAULT_API_BASE;
}
backendUrl.addEventListener('change', () => {
    try { localStorage.setItem('hp_backend', backendUrl.value.trim()); } catch (_) {}
    pingBackend();
});

async function pingBackend() {
    backendDot.className = 'backend-dot';
    try {
        // FIXED: Hits the backend index info endpoint to safely check health state
        const res = await fetch(`${getApiBase()}/api/health`, { headers: NGROK_HEADERS });
        backendDot.className = res.ok ? 'backend-dot ok' : 'backend-dot bad';
    } catch (_) {
        try {
            // Fallback to base root route if explicit health subpath returns 404
            const resFallback = await fetch(`${getApiBase()}/`, { headers: NGROK_HEADERS });
            backendDot.className = resFallback.ok ? 'backend-dot ok' : 'backend-dot bad';
        } catch (__) {
            backendDot.className = 'backend-dot bad';
        }
    }
}

/* ── confidence slider ────────────────────────────────────────────────── */
confSlider.addEventListener('input', () => {
    const v = confSlider.value;
    confVal.textContent = (v / 100).toFixed(2);
    confSlider.style.setProperty('--pct', v);
});

/* ── drag & drop / file pick ──────────────────────────────────────────── */
dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('over');
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(f) {
    if (!f.type.startsWith('image/')) { showError('Please choose an image file.'); return; }
    selectedFile = f;
    analyzeBtn.disabled = false;
    dropPreview.src = URL.createObjectURL(f);
    previewWrap.style.display = 'block';
    dropPrompt.style.display  = 'none';
    hideError();
    results.classList.remove('active');
}

/* ── the decipherment ritual (loading) ────────────────────────────────── */
function stages() { return [...ritual.querySelectorAll('.stage')]; }

function resetRitual() {
    stages().forEach((s) => s.classList.remove('active', 'done'));
}
function startRitual() {
    resetRitual();
    ritual.classList.add('active');
    dropZone.classList.add('reading');
    let i = 0;
    const list = stages();
    list[0].classList.add('active');
    ritualTimer = setInterval(() => {
        if (i < RITUAL_STAGES - 1) {
            list[i].classList.remove('active');
            list[i].classList.add('done');
            i++;
            list[i].classList.add('active');
        }
    }, STAGE_MS);
}
function stopRitual() {
    clearInterval(ritualTimer);
    stages().forEach((s) => { s.classList.remove('active'); s.classList.add('done'); });
    dropZone.classList.remove('reading');
    setTimeout(() => ritual.classList.remove('active'), 450);
}

/* ── analyse ──────────────────────────────────────────────────────────── */
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
    if (!selectedFile) return;
    hideError();
    results.classList.remove('active');
    analyzeBtn.disabled = true;
    startRitual();
    const startedAt = Date.now();

    const fd = new FormData();
    fd.append('image', selectedFile);
    fd.append('conf_thresh', (confSlider.value / 100).toFixed(2));

    try {
        // FIXED: Adjusted response handling to align with Flask server API routing structure
        const res = await fetch(`${getApiBase()}/api/analyze`, {
            method: 'POST',
            headers: NGROK_HEADERS,
            body: fd,
        });

        let payload;
        try { payload = await res.json(); }
        catch (_) { throw new Error(`Bad response (HTTP ${res.status}).`); }

        if (!res.ok || !payload.success) {
            throw new Error(payload?.error?.message || payload?.error || `HTTP ${res.status}`);
        }

        const wait = Math.max(0, MIN_RITUAL_MS - (Date.now() - startedAt));
        setTimeout(() => { stopRitual(); renderResults(payload.data); }, wait);
        backendDot.className = 'backend-dot ok';

    } catch (err) {
        stopRitual();
        showError(err.message || 'Network error.');
        backendDot.className = 'backend-dot bad';
    } finally {
        analyzeBtn.disabled = false;
    }
}

/* ── render ───────────────────────────────────────────────────────────── */
function sentimentClass(label) {
    const s = (label || '').toLowerCase();
    if (s === 'positive') return 'positive';
    if (s === 'negative') return 'negative';
    return 'neutral';
}
function sentimentEmoji(label) {
    const s = (label || '').toLowerCase();
    if (s === 'positive') return '😊';
    if (s === 'negative') return '😞';
    return '😐';
}
function titleCase(str) {
    return (str || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
function pct(x) { return `${Math.round((x || 0) * 100)}%`; }

function renderResults(d) {
    // 1) The returned image (Uses clean original image data payload from Flask endpoint)
    $('resultImg').src = d.image || '';

    // 2) sign-count badge + reading direction
    const n   = d.n_signs || 0;
    const dir = d.direction ? ` · ${d.direction.toUpperCase()}` : '';
    $('signBadge').textContent =
        n === 0 ? 'no glyphs found' : `${n} sign${n === 1 ? '' : 's'}${dir}`;

    // 3) English + Arabic
    $('englishResult').textContent = d.english || '—';
    $('arabicResult').textContent  = d.arabic  || '—';

    // 4) sentiment badge
    const label = d.sentiment || 'Neutral';
    $('sentimentResult').innerHTML =
        `<span class="badge ${sentimentClass(label)}">${sentimentEmoji(label)} ${label}` +
        `<span class="badge-pct">${pct(d.sent_confidence)}</span></span>`;

    // 5) intention (EN label + Arabic + confidence + ambiguity flag)
    const amb = d.intent_ambiguous ? ' <span class="intent-amb">⚠︎</span>' : '';
    $('intentionEn').innerHTML = d.intention_en
        ? `🎯 ${titleCase(d.intention_en)} <span class="badge-pct">(${pct(d.intent_confidence)})</span>${amb}`
        : '—';
    $('intentionAr').textContent = d.intention_ar || '';

    results.classList.add('active');
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── error helpers ────────────────────────────────────────────────────── */
function showError(msg) {
    errorMsg.textContent = msg;
    errorBox.classList.add('show');
}
function hideError() { errorBox.classList.remove('show'); }

/* ── keyboard: Enter on the drop zone opens the picker ────────────────── */
dropZone.setAttribute('tabindex', '0');
dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

/* ── init ─────────────────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
    loadSavedBackend();
    pingBackend();
    setInitialHints();
});

function setInitialHints() {
    $('englishResult').innerHTML  = '<span class="hint">upload a photo and press Decipher</span>';
    $('arabicResult').innerHTML   = '<span class="hint">ارفع صورة واضغط Decipher</span>';
    $('sentimentResult').innerHTML = '<span class="hint">tone appears after deciphering</span>';
    $('intentionEn').innerHTML    = '<span class="hint">intention appears after deciphering</span>';
    $('intentionAr').textContent  = '';
}