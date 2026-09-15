/* app.js */
document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatHistory = document.getElementById('chatHistory');
    const apiKeyInput = document.getElementById('apiKey');
    const newChatBtn = document.getElementById('newChatBtn');
    const sendBtn = document.getElementById('sendBtn');

    const API_BASE = 'http://localhost:9000';

    // node_name from the backend -> what the user should read
    const STEP_LABELS = {
        history_summarizer_node: 'Summarised earlier messages',
        precheck_node: 'Read your question',
        orchestrator_node: 'Gathered market data',
        answer_node: 'Wrote the answer',
    };

    let currentConversationId = null;
    let activeStream = null; // AbortController of the turn in flight

    // Load API Key from localStorage
    const savedApiKey = localStorage.getItem('finagent_api_key');
    if (savedApiKey) {
        apiKeyInput.value = savedApiKey;
    }

    apiKeyInput.addEventListener('input', (e) => {
        localStorage.setItem('finagent_api_key', e.target.value);
    });

    newChatBtn.addEventListener('click', () => {
        // Drop the in-flight stream, otherwise its events land in a cleared view.
        if (activeStream) activeStream.abort();
        currentConversationId = null;
        chatHistory.innerHTML = `
            <div class="message system-msg">
                <div class="msg-content">Started a new conversation.</div>
            </div>
        `;
    });

    /* ---------------------------------------------------------------- SSE */

    /**
     * EventSource can't do POST or custom headers, so we read the
     * text/event-stream body off fetch() and parse the wire format here.
     * Events are separated by a blank line; lines starting with ':' are
     * comments (that's the server's keep-alive ping).
     */
    async function* readSSE(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');

                let sep;
                while ((sep = buffer.indexOf('\n\n')) !== -1) {
                    const block = buffer.slice(0, sep);
                    buffer = buffer.slice(sep + 2);
                    const parsed = parseEventBlock(block);
                    if (parsed) yield parsed;
                }
            }
        } finally {
            reader.cancel().catch(() => {});
        }
    }

    function parseEventBlock(block) {
        let event = 'message';
        const dataLines = [];

        for (const rawLine of block.split('\n')) {
            const line = rawLine.replace(/\r$/, '');
            if (!line || line.startsWith(':')) continue; // blank or keep-alive
            const colon = line.indexOf(':');
            const field = colon === -1 ? line : line.slice(0, colon);
            let value = colon === -1 ? '' : line.slice(colon + 1);
            if (value.startsWith(' ')) value = value.slice(1);

            if (field === 'event') event = value;
            else if (field === 'data') dataLines.push(value);
        }

        // Per spec an event with an empty data buffer is never dispatched.
        if (!dataLines.length) return null;

        const raw = dataLines.join('\n');
        try {
            return { event, data: JSON.parse(raw) };
        } catch {
            return { event, data: raw };
        }
    }

    async function openStream(message, apiKey, signal) {
        const url = currentConversationId
            ? `${API_BASE}/stream/conversation/${encodeURIComponent(currentConversationId)}`
            : `${API_BASE}/stream/conversation`;

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Accept: 'text/event-stream',
                'x-api-key': apiKey,
            },
            body: JSON.stringify({ conversation: message }),
            signal,
        });

        // Auth and rate-limit rejections come back as JSON, not as a stream.
        if (!response.ok) {
            const body = await response.json().catch(() => ({}));
            throw new Error(body.detail || body.error || `Server error: ${response.status}`);
        }
        return response;
    }

    /* ------------------------------------------------------------------ UI */

    function appendMessage(role, content, steps) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-msg`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'msg-content';
        contentDiv.textContent = content;
        msgDiv.appendChild(contentDiv);

        if (steps && steps.length) {
            contentDiv.appendChild(buildTrace(steps));
        }

        chatHistory.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function appendProgress() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant-msg progress-msg';
        msgDiv.innerHTML = `
            <div class="msg-content">
                <div class="progress-steps"></div>
                <div class="progress-current">
                    <div class="typing-indicator">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                    <span class="progress-label">Working…</span>
                </div>
            </div>
        `;
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function addStepRow(progressEl, step) {
        const failed = step.status === 'failed';
        const row = document.createElement('div');
        row.className = `step-row ${failed ? 'failed' : 'ok'}`;

        const mark = document.createElement('span');
        mark.className = 'step-mark';
        mark.textContent = failed ? '✕' : '✓';

        const name = document.createElement('span');
        name.className = 'step-name';
        name.textContent = STEP_LABELS[step.node] || step.node;

        const time = document.createElement('span');
        time.className = 'step-time';
        time.textContent = formatMs(step.duration_ms);

        row.append(mark, name, time);
        progressEl.querySelector('.progress-steps').appendChild(row);
        scrollToBottom();
    }

    function buildTrace(steps) {
        const total = steps.reduce((sum, s) => sum + (s.duration_ms || 0), 0);
        const details = document.createElement('details');
        details.className = 'trace';

        const summary = document.createElement('summary');
        summary.textContent = `${steps.length} step${steps.length === 1 ? '' : 's'} · ${formatMs(total)}`;
        details.appendChild(summary);

        for (const step of steps) {
            const row = document.createElement('div');
            row.className = `step-row ${step.status === 'failed' ? 'failed' : 'ok'}`;

            const name = document.createElement('span');
            name.className = 'step-name';
            name.textContent = STEP_LABELS[step.node] || step.node;

            const time = document.createElement('span');
            time.className = 'step-time';
            time.textContent = formatMs(step.duration_ms);

            row.append(name, time);
            details.appendChild(row);
        }
        return details;
    }

    function removeElement(element) {
        if (element && element.parentNode) {
            element.parentNode.removeChild(element);
        }
    }

    function formatMs(ms) {
        if (typeof ms !== 'number') return '';
        return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
    }

    function setBusy(busy) {
        chatInput.disabled = busy;
        sendBtn.disabled = busy;
        if (!busy) chatInput.focus();
    }

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    /* ------------------------------------------------------------- the turn */

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const message = chatInput.value.trim();
        const apiKey = apiKeyInput.value.trim();

        if (!message) return;

        if (!apiKey) {
            alert('Please enter your API Key in the sidebar first.');
            apiKeyInput.focus();
            return;
        }

        chatInput.value = '';
        setBusy(true);
        appendMessage('user', message);

        const progressEl = appendProgress();
        const steps = [];
        const controller = new AbortController();
        activeStream = controller;

        try {
            const response = await openStream(message, apiKey, controller.signal);
            let terminal = null;

            for await (const { event, data } of readSSE(response)) {
                if (event === 'step') {
                    steps.push(data);
                    addStepRow(progressEl, data);
                } else if (event === 'done' || event === 'error') {
                    terminal = { event, data };
                }
            }

            removeElement(progressEl);

            if (!terminal) {
                appendMessage('system', 'Connection closed before the answer arrived.');
            } else {
                if (terminal.data.conversation_id) {
                    currentConversationId = terminal.data.conversation_id;
                }
                if (terminal.event === 'done') {
                    appendMessage('assistant', terminal.data.conversation || 'No response content.', steps);
                } else {
                    appendMessage('system', terminal.data.conversation || 'The run failed.');
                }
            }
        } catch (error) {
            removeElement(progressEl);
            if (error.name !== 'AbortError') {
                appendMessage('system', `Error: ${error.message}`);
            }
        } finally {
            activeStream = null;
            setBusy(false);
        }
    });
});
