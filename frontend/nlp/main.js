// const API_BASE = 'http://localhost:5000';
const API_BASE = 'https://irretrievably-unsimpering-darrin.ngrok-free.dev';

let examples = {};

function showError(message) {
    const box = document.getElementById('errorBox');
    box.textContent = `❌ Error: ${message}`;
    box.classList.remove('hidden');
    setTimeout(() => box.classList.add('hidden'), 5000);
}

function showLoading(btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Processing...';
}

function hideLoading(btn) {
    btn.disabled = false;
    btn.innerHTML = '⟡ &nbsp; Decipher';
}

function getSentimentEmoji(sentiment) {
    if (!sentiment) return '😐';
    const s = sentiment.toLowerCase();
    if (s === 'positive') return '😊';
    else if (s === 'negative') return '😞';
    else return '😐';
}

function updateGlyphPreview(glyphStr) {
    const box   = document.getElementById('glyphPreviewBox');
    const inner = document.getElementById('glyphPreviewGlyphs');
    if (glyphStr === 'loading') {
        inner.innerHTML = '<span class="glyph-preview-placeholder">Loading<span class="loading-dots"></span></span>';
        box.classList.remove('active');
    }
    else if (glyphStr && glyphStr.trim()) {
        inner.innerHTML = glyphStr;
        box.classList.add('active');
    }
    else {
        inner.innerHTML = '<span class="glyph-preview-placeholder">Enter codes and press Decipher to reveal glyphs</span>';
        box.classList.remove('active');
    }
}

async function loadExamples() {
    try {
        const response = await fetch(`${API_BASE}/api/examples`, {
            headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        const data = await response.json();
        examples = data;
        renderExamples();
    } catch (e) {
        console.warn('Could not load examples:', e);
        document.getElementById('examplesContainer').innerHTML = '';
    }
}

function renderExamples() {
    const container = document.getElementById('examplesContainer');
    container.innerHTML = '';
    for (const [key, example] of Object.entries(examples)) {
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.title = example.description;
        // backend returns codes as a string e.g. "G17 M18 F34"
        const codesDisplay = Array.isArray(example.codes)
            ? example.codes.join(', ')
            : example.codes;
        chip.textContent = codesDisplay;
        chip.onclick = () => {
            document.getElementById('codesInput').value = codesDisplay;
            document.getElementById('codesInput').focus();
        };
        container.appendChild(chip);
    }
}

async function decipher() {
    const input = document.getElementById('codesInput').value.trim();
    if (!input) {
        showError('Please enter Gardiner codes');
        return;
    }

    // Build the gardiner string the backend expects (space-separated, uppercase)
    const gardiner = input
        .split(/[\s,]+/)
        .map(c => c.trim().toUpperCase())
        .filter(c => c)
        .join(' ');

    if (!gardiner) {
        showError('Invalid codes format');
        return;
    }

    const btn = document.getElementById('decipherBtn');
    showLoading(btn);
    clearResults();

    try {
        const response = await fetch(`${API_BASE}/api/decipher`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify({ gardiner })   // ✅ field name the backend expects
        });

        if (!response.ok) {
            let errMsg = `HTTP ${response.status}`;
            try {
                const error = await response.json();
                errMsg = error?.error?.message || error?.error || errMsg;
            } catch (_) {}
            throw new Error(errMsg);
        }

        const result = await response.json();
        if (result.success) {
            renderResults(result.data);
        } else {
            showError(result.error?.message || result.error || 'Unknown error');
        }
    } catch (e) {
        showError(e.message);
    } finally {
        hideLoading(btn);
    }
}

function clearResults() {
    updateGlyphPreview('loading');
    document.getElementById('signTableBody').innerHTML = '';
    const hint = document.getElementById('tableHint');
    hint.style.display = 'block';
    hint.textContent   = 'Loading...';
    document.getElementById('englishResult').textContent  = '';
    document.getElementById('arabicResult').textContent   = '';
    document.getElementById('sentimentResult').innerHTML  = '';
    document.getElementById('intentionEn').textContent    = '';
    document.getElementById('intentionAr').textContent    = '';
}

// Build one <tr> for the SIGN ANALYSIS table
function buildSignRow(sign) {
    const tr = document.createElement('tr');

    const tdCode = document.createElement('td');
    tdCode.className = 'sign-code';
    tdCode.textContent = sign.code || '';

    const tdGlyph = document.createElement('td');
    tdGlyph.className = 'sign-glyph';
    tdGlyph.innerHTML = (sign.glyph && sign.glyph.trim())
        ? `<span class="glyph">${sign.glyph}</span>`
        : '<span class="glyph-missing">—</span>';

    const tdPhon = document.createElement('td');
    tdPhon.className = 'sign-phonetic';
    tdPhon.textContent = (sign.phonetic && sign.phonetic.trim()) ? sign.phonetic : '—';

    const tdMean = document.createElement('td');
    tdMean.className = 'sign-meaning';
    tdMean.textContent = sign.meaning || '';

    tr.appendChild(tdCode);
    tr.appendChild(tdGlyph);
    tr.appendChild(tdPhon);
    tr.appendChild(tdMean);
    return tr;
}

function renderResults(data) {
    // 1) Big glyph preview box — glyphs come from the backend (xlsx sign list)
    updateGlyphPreview(data.glyphs || '');

    // 2) SIGN ANALYSIS table — code / glyph / phonetic(from Stage 1) / meaning
    const tbody = document.getElementById('signTableBody');
    const hint  = document.getElementById('tableHint');
    const rows  = Array.isArray(data.sign_analysis) ? data.sign_analysis : [];

    tbody.innerHTML = '';
    if (rows.length) {
        hint.style.display = 'none';
        for (const sign of rows) {
            tbody.appendChild(buildSignRow(sign));
        }
    } else {
        // fallback: no per-sign rows -> show phonetics / assembled words
        if (data.spaced_phonetics || data.assembled_words) {
            hint.style.display = 'block';
            hint.innerHTML =
                (data.spaced_phonetics
                    ? `<span class="phonetic-line">🔤 ${data.spaced_phonetics}</span><br>`
                    : '') +
                (data.assembled_words
                    ? `<span class="assembled-line">📝 ${data.assembled_words}</span>`
                    : '');
        } else {
            hint.style.display = 'none';
        }
    }

    // 3) Translations
    document.getElementById('englishResult').textContent = data.english || '—';
    document.getElementById('arabicResult').textContent  = data.arabic  || '—';

    // 4) Sentiment
    const emoji = getSentimentEmoji(data.sentiment);
    const sentScore = data.sent_score ? ` (${data.sent_score})` : '';
    document.getElementById('sentimentResult').innerHTML =
        `<span class="sent-badge">${emoji} ${data.sentiment || 'Neutral'}${sentScore}</span>`;

    // 5) Intention — English label + Arabic translation
    document.getElementById('intentionEn').textContent =
        data.intention_en ? `🎯 ${data.intention_en}` : '—';
    document.getElementById('intentionAr').textContent =
        data.intention_ar ? data.intention_ar : '';
}

document.getElementById('decipherBtn').addEventListener('click', decipher);
document.getElementById('codesInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') decipher();
});

window.addEventListener('DOMContentLoaded', () => {
    loadExamples();
    setInitialHints();
});

function setInitialHints() {
    document.getElementById('englishResult').innerHTML =
        '<span class="card-hint">press Decipher to reveal the translation</span>';
    document.getElementById('arabicResult').innerHTML =
        '<span class="card-hint">اضغط Decipher لعرض الترجمة</span>';
    document.getElementById('sentimentResult').innerHTML =
        '<span class="card-hint">press Decipher to analyse the tone</span>';
    document.getElementById('intentionEn').innerHTML =
        '<span class="card-hint">press Decipher to detect intention</span>';
    document.getElementById('intentionAr').innerHTML = '';
}