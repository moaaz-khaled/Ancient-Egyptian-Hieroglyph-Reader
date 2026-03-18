const API_BASE = 'http://localhost:5000';
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
    if (sentiment === 'positive') return '😊';
    if (sentiment === 'negative') return '😞';
    return '😐';
}


async function loadExamples() {
    try {
        const response = await fetch(`${API_BASE}/api/examples`);
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
        chip.textContent = example.codes.join(', ');
        chip.onclick = () => {
            document.getElementById('codesInput').value = example.codes.join(', ');
            decipher();
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

    const codes = input.split(',').map(c => c.trim().toUpperCase()).filter(c => c);
    if (codes.length === 0) {
        showError('Invalid codes format');
        return;
    }

    const btn = document.getElementById('decipherBtn');
    showLoading(btn);
    clearResults();

    try {
        const response = await fetch(`${API_BASE}/api/decipher`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ codes })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'API Error');
        }

        const result = await response.json();
        if (result.success) {
            renderResults(result.data);
        } else {
            showError(result.error);
        }
    } catch (e) {
        showError(e.message);
    } finally {
        hideLoading(btn);
    }
}


function clearResults() 
{
    document.getElementById('signTableBody').innerHTML  = '';
    const hint = document.getElementById('tableHint');
    hint.style.display  = 'block';
    hint.textContent    = 'Loading...';
    document.getElementById('englishResult').textContent  = '';
    document.getElementById('arabicResult').textContent   = '';
    document.getElementById('sentimentResult').innerHTML  = '';
    document.getElementById('intentionEn').textContent    = '';
    document.getElementById('intentionAr').textContent    = '';
}


function renderResults(data) 
{
    document.getElementById('tableHint').style.display = 'none';

    const tbody      = document.getElementById('signTableBody');
    tbody.innerHTML  = '';
    const glyphParts = data.glyphs ? data.glyphs.split(' ') : [];

    for (let i = 0; i < data.per_sign.length; i++) 
    {
        const [code, phonetic, meaning] = data.per_sign[i];
        const row      = document.createElement('tr');
        const glyphStr = glyphParts[i] || '□';
        row.innerHTML  = `
            <td><strong>${code.toUpperCase()}</strong></td>
            <td class="glyph-cell">${glyphStr}</td>
            <td>${phonetic || '(det.)'}</td>
            <td>${meaning  || '-'}</td>
        `;
        tbody.appendChild(row);
    }

    const displayEnglish = (data.sentence && data.sentence.length > 0)
        ? data.sentence : data.english;
    document.getElementById('englishResult').textContent = displayEnglish || '';
    document.getElementById('arabicResult').textContent  = data.arabic    || '';

    const emoji = getSentimentEmoji(data.sentiment);
    document.getElementById('sentimentResult').innerHTML =
        `<span class="sent-badge">${emoji} ${data.sentiment} &nbsp; (${data.sent_score})</span>`;

    document.getElementById('intentionEn').textContent =
        data.intention_en ? `🎯 ${data.intention_en}` : '';
    document.getElementById('intentionAr').textContent = data.intention_ar || '';
}


document.getElementById('decipherBtn').addEventListener('click', decipher);
document.getElementById('codesInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') decipher();
});

window.addEventListener('DOMContentLoaded', () => {
    loadExamples();
    setInitialHints();
});


function setInitialHints() 
{
    document.getElementById('englishResult').innerHTML   =
        '<span class="card-hint">press Decipher to reveal the translation</span>';
    document.getElementById('arabicResult').innerHTML    =
        '<span class="card-hint">اضغط Decipher لعرض الترجمة</span>';
    document.getElementById('sentimentResult').innerHTML =
        '<span class="card-hint">press Decipher to analyse the tone</span>';
    document.getElementById('intentionEn').innerHTML     =
        '<span class="card-hint">press Decipher to detect intention</span>';
}