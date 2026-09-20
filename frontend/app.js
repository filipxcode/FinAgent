/* app.js */
document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatHistory = document.getElementById('chatHistory');
    const apiKeyInput = document.getElementById('apiKey');
    const newChatBtn = document.getElementById('newChatBtn');
    const sendBtn = document.getElementById('sendBtn');

    const API_BASE = 'http://localhost:9000';

    // node name from the backend -> what the user should read while it runs / once done
    const STEP_LABELS = {
        history_summarizer_node: { running: 'Summarising earlier messages…', done: 'Summarised earlier messages' },
        precheck_node: { running: 'Reading your question…', done: 'Read your question' },
        orchestrator_node: { running: 'Gathering market data…', done: 'Gathered market data' },
        answer_node: { running: 'Writing the answer…', done: 'Wrote the answer' },
    };
    // Specialist agents the orchestrator hands work to.
    const DELEGATION_LABELS = {
        researcher: { running: 'Researching…', done: 'Researched', failed: 'Research failed' },
        whale_tracker: {
            running: 'Checking whale activity…',
            done: 'Checked whale activity',
            failed: 'Whale activity check failed',
        },
        news_agent: { running: 'Reading the news…', done: 'Read the news', failed: 'News lookup failed' },
    };
    const MARKS = { running: '', finished: '✓', failed: '✕' };

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

    /* A step is { name, status, duration_ms, delegations: [{ id, agent, task, status }] }
       with status running | finished | failed. The live view and the trace kept
       under the answer both render from this one shape. */

    function labelFor(labels, key, status) {
        const entry = labels[key];
        if (!entry) return key;
        if (status === 'running') return entry.running;
        if (status === 'failed' && entry.failed) return entry.failed;
        return entry.done;
    }

    function createRow(className) {
        const el = document.createElement('div');
        el.className = `step-row ${className || ''}`.trim();

        const mark = document.createElement('span');
        mark.className = 'step-mark';

        const name = document.createElement('span');
        name.className = 'step-name';

        const detail = document.createElement('span');
        detail.className = 'step-detail';

        const time = document.createElement('span');
        time.className = 'step-time';

        el.append(mark, name, detail, time);

        return {
            el,
            paint({ status, label, detailText, timeText }) {
                el.classList.remove('running', 'finished', 'failed');
                el.classList.add(status);
                mark.textContent = MARKS[status];
                name.textContent = label;
                detail.textContent = detailText || '';
                detail.hidden = !detailText;
                time.textContent = timeText || '';
            },
        };
    }

    function createStepGroup(step) {
        const el = document.createElement('div');
        el.className = 'step-group';

        const row = createRow();
        const list = document.createElement('div');
        list.className = 'delegations';
        el.append(row.el, list);

        const delegationRows = new Map();

        function render() {
            row.paint({
                status: step.status,
                label: labelFor(STEP_LABELS, step.name, step.status),
                timeText: step.status === 'running' ? '' : formatMs(step.duration_ms),
            });
            for (const d of step.delegations) {
                let dRow = delegationRows.get(d.id);
                if (!dRow) {
                    dRow = createRow('delegation-row');
                    delegationRows.set(d.id, dRow);
                    list.appendChild(dRow.el);
                }
                dRow.paint({
                    status: d.status,
                    label: labelFor(DELEGATION_LABELS, d.agent, d.status),
                    detailText: d.task,
                });
            }
        }

        render();
        return { el, render };
    }

    /* Folds the SSE events of one turn into steps and keeps `container` in sync. */
    function createTurnTracker(container) {
        const steps = [];
        const groups = new Map();

        function ensureStep(name) {
            let step = steps.find((s) => s.name === name);
            if (!step) {
                step = { name, status: 'running', duration_ms: null, delegations: [] };
                steps.push(step);
                const group = createStepGroup(step);
                groups.set(name, group);
                container.appendChild(group.el);
            }
            return step;
        }

        function handle(event, data) {
            if (event === 'step.started') {
                ensureStep(data.step);
            } else if (event === 'step.finished') {
                const step = ensureStep(data.step);
                step.status = data.status;
                step.duration_ms = data.duration_ms;
                // A delegation still running when its step ends was cut short.
                for (const d of step.delegations) {
                    if (d.status === 'running') d.status = 'failed';
                }
                groups.get(step.name).render();
            } else if (event === 'delegation') {
                const step = data.step ? ensureStep(data.step) : steps[steps.length - 1];
                if (!step) return;
                const status = data.status === 'started' ? 'running' : data.status;
                const known = step.delegations.find((d) => d.id === data.id);
                if (known) {
                    known.status = status;
                } else {
                    step.delegations.push({ id: data.id, agent: data.agent, task: data.task, status });
                }
                groups.get(step.name).render();
            } else {
                return;
            }
            scrollToBottom();
        }

        return { steps, handle };
    }

    function buildTrace(steps) {
        const total = steps.reduce((sum, s) => sum + (s.duration_ms || 0), 0);
        const delegated = steps.reduce((sum, s) => sum + s.delegations.length, 0);
        const details = document.createElement('details');
        details.className = 'trace';

        const parts = [`${steps.length} step${steps.length === 1 ? '' : 's'}`];
        if (delegated) parts.push(`${delegated} specialist call${delegated === 1 ? '' : 's'}`);
        parts.push(formatMs(total));

        const summary = document.createElement('summary');
        summary.textContent = parts.join(' · ');
        details.appendChild(summary);

        for (const step of steps) {
            details.appendChild(createStepGroup(step).el);
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
        const tracker = createTurnTracker(progressEl.querySelector('.progress-steps'));
        const controller = new AbortController();
        activeStream = controller;

        try {
            const response = await openStream(message, apiKey, controller.signal);
            let terminal = null;

            for await (const { event, data } of readSSE(response)) {
                if (event === 'done' || event === 'error') {
                    terminal = { event, data };
                } else {
                    tracker.handle(event, data);
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
                    appendMessage('assistant', terminal.data.conversation || 'No response content.', tracker.steps);
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
