/* app.js */
document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatHistory = document.getElementById('chatHistory');
    const apiKeyInput = document.getElementById('apiKey');
    const newChatBtn = document.getElementById('newChatBtn');
    const chatList = document.getElementById('chatList');
    const chatTitle = document.getElementById('chatTitle');
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

    const ACTIVE_CHAT_KEY = 'finagent_active_chat';
    const CONVERSATIONS_PAGE_SIZE = 30;
    const HISTORY_PAGE_SIZE = 50;
    const TITLE_MAX_LENGTH = 40;

    // The chat list lives on the server (GET /conversations); only the open chat is remembered here.
    let chats = []; // [{ id, title, updatedAt, local? }], most recently active first
    let chatsCursor = null; // `before` value for the next page of the list, null = no more
    let chatsStatus = 'idle'; // idle | loading | error
    let chatsError = '';
    let currentConversationId = localStorage.getItem(ACTIVE_CHAT_KEY); // null = a fresh chat with no id yet
    let activeStream = null; // AbortController of the turn in flight
    let viewSeq = 0; // bumped on every chat switch so late responses for a previous chat are dropped
    let listSeq = 0; // same guard for the chat list

    // Load API Key from localStorage
    const savedApiKey = localStorage.getItem('finagent_api_key');
    if (savedApiKey) {
        apiKeyInput.value = savedApiKey;
    }

    apiKeyInput.addEventListener('input', (e) => {
        localStorage.setItem('finagent_api_key', e.target.value);
    });

    // A key entered after the page restored a chat: load what was waiting for it.
    apiKeyInput.addEventListener('change', () => {
        loadConversations();
        if (apiKeyInput.value.trim() && currentConversationId && !activeStream) {
            openChat(currentConversationId);
        }
    });

    newChatBtn.addEventListener('click', showNewChat);

    /* --------------------------------------------------------------- chats */

    function persistActiveChat() {
        if (currentConversationId) localStorage.setItem(ACTIVE_CHAT_KEY, currentConversationId);
        else localStorage.removeItem(ACTIVE_CHAT_KEY);
    }

    function newConversationId() {
        return typeof crypto !== 'undefined' && crypto.randomUUID
            ? crypto.randomUUID()
            : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    }

    function titleFrom(message) {
        const oneLine = message.replace(/\s+/g, ' ').trim();
        return oneLine.length > TITLE_MAX_LENGTH ? `${oneLine.slice(0, TITLE_MAX_LENGTH - 1)}…` : oneLine;
    }

    /* Register the chat on first use and float it to the top of the list. */
    function touchChat(id, firstMessage) {
        const existing = chats.find((c) => c.id === id);
        const chat = existing || { id, title: titleFrom(firstMessage), local: true };
        chat.updatedAt = new Date().toISOString();
        chats = [chat, ...chats.filter((c) => c.id !== id)];
        persistActiveChat();
        renderChatList();
        renderHeader();
    }

    function renderHeader() {
        const chat = chats.find((c) => c.id === currentConversationId);
        if (chat) chatTitle.textContent = chat.title;
        else chatTitle.textContent = currentConversationId ? 'Conversation' : 'New conversation';
    }

    function renderChatList() {
        chatList.replaceChildren();

        if (chatsStatus === 'error') {
            const failed = document.createElement('div');
            failed.className = 'chat-list-empty';
            failed.textContent = `Could not load chats: ${chatsError}`;
            const retry = document.createElement('button');
            retry.type = 'button';
            retry.className = 'load-older';
            retry.textContent = 'Retry';
            retry.addEventListener('click', () => loadConversations());
            chatList.append(failed, retry);
        } else if (!chats.length) {
            const empty = document.createElement('div');
            empty.className = 'chat-list-empty';
            empty.textContent = chatsStatus === 'loading'
                ? 'Loading chats…'
                : 'No chats yet. Send a message to start one.';
            chatList.appendChild(empty);
        }

        for (const chat of chats) {
            const item = document.createElement('div');
            item.className = `chat-item${chat.id === currentConversationId ? ' active' : ''}`;

            const open = document.createElement('button');
            open.type = 'button';
            open.className = 'chat-item-title';
            open.textContent = chat.title;
            open.title = chat.title;
            if (chat.id === currentConversationId) open.setAttribute('aria-current', 'true');
            open.addEventListener('click', () => {
                if (chat.id !== currentConversationId) openChat(chat.id);
            });

            item.appendChild(open);
            chatList.appendChild(item);
        }

        if (chatsCursor && chatsStatus !== 'error') {
            const more = document.createElement('button');
            more.type = 'button';
            more.className = 'load-older';
            more.textContent = chatsStatus === 'loading' ? 'Loading…' : 'Show older chats';
            more.disabled = chatsStatus === 'loading';
            more.addEventListener('click', () => loadConversations(true));
            chatList.appendChild(more);
        }
    }

    function leaveCurrentView() {
        // Drop the in-flight stream, otherwise its events land in another chat's view.
        if (activeStream) activeStream.abort();
        return ++viewSeq;
    }

    function showNewChat() {
        leaveCurrentView();
        currentConversationId = null;
        persistActiveChat();
        chatHistory.replaceChildren(buildMessage('system', 'Started a new conversation.'));
        renderChatList();
        renderHeader();
        chatInput.focus();
    }

    async function openChat(id) {
        const seq = leaveCurrentView();
        currentConversationId = id;
        persistActiveChat();
        renderChatList();
        renderHeader();

        if (!apiKeyInput.value.trim()) {
            chatHistory.replaceChildren(buildMessage('system', 'Enter your API key in the sidebar to load this chat.'));
            return;
        }

        const loading = buildMessage('system', 'Loading messages…');
        chatHistory.replaceChildren(loading);
        try {
            const page = await fetchHistory(id);
            if (seq !== viewSeq) return;
            if (!page.messages.length) {
                chatHistory.replaceChildren(buildMessage('system', 'No messages in this chat yet.'));
                return;
            }
            chatHistory.replaceChildren(historyFragment(page, seq));
            scrollToBottom();
        } catch (error) {
            if (seq !== viewSeq) return;
            chatHistory.replaceChildren(buildMessage('system', `Could not load this chat: ${error.message}`));
        }
    }

    async function apiGet(path, params) {
        const response = await fetch(`${API_BASE}${path}?${new URLSearchParams(params)}`, {
            headers: { 'x-api-key': apiKeyInput.value.trim() },
        });
        if (!response.ok) {
            const body = await response.json().catch(() => ({}));
            throw new Error(body.detail || body.error || `Server error: ${response.status}`);
        }
        return response.json();
    }

    function fetchHistory(id, before) {
        // URLSearchParams escapes the '+' of the UTC offset in the cursor.
        const params = { conversation_id: id, limit: String(HISTORY_PAGE_SIZE) };
        if (before) params.before = before;
        return apiGet('/history', params);
    }

    /* First page replaces the list; `more` appends the page after the current one. */
    async function loadConversations(more = false) {
        if (!apiKeyInput.value.trim()) {
            chats = chats.filter((c) => c.local);
            chatsCursor = null;
            chatsStatus = 'idle';
            renderChatList();
            return;
        }

        const seq = ++listSeq;
        chatsStatus = 'loading';
        renderChatList();
        try {
            const params = { limit: String(CONVERSATIONS_PAGE_SIZE) };
            if (more && chatsCursor) params.before = chatsCursor;
            const page = await apiGet('/conversations', params);
            if (seq !== listSeq) return;

            const incoming = page.conversations.map((c) => ({
                id: c.conversation_id,
                title: c.title,
                updatedAt: c.updated_at,
            }));
            const known = new Set(incoming.map((c) => c.id));
            if (more) {
                const have = new Set(chats.map((c) => c.id));
                chats = [...chats, ...incoming.filter((c) => !have.has(c.id))];
            } else {
                // A chat started here that the server has not listed yet stays on top.
                chats = [...chats.filter((c) => c.local && !known.has(c.id)), ...incoming];
            }
            chatsCursor = page.next_cursor;
            chatsStatus = 'idle';
        } catch (error) {
            if (seq !== listSeq) return;
            chatsStatus = 'error';
            chatsError = error.message;
        }
        renderChatList();
        renderHeader();
    }

    function historyFragment(page, seq) {
        const fragment = document.createDocumentFragment();

        if (page.next_cursor) {
            const older = document.createElement('button');
            older.type = 'button';
            older.className = 'load-older';
            older.textContent = 'Load earlier messages';
            older.addEventListener('click', () => loadOlder(older, page.next_cursor, seq));
            fragment.appendChild(older);
        }
        for (const message of page.messages) {
            fragment.appendChild(buildMessage(message.role, message.content));
        }
        return fragment;
    }

    async function loadOlder(button, cursor, seq) {
        button.disabled = true;
        button.textContent = 'Loading…';
        try {
            const page = await fetchHistory(currentConversationId, cursor);
            if (seq !== viewSeq) return;

            // Keep the message the reader was looking at where it is.
            const heightBefore = chatHistory.scrollHeight;
            const topBefore = chatHistory.scrollTop;
            button.replaceWith(historyFragment(page, seq));
            chatHistory.style.scrollBehavior = 'auto';
            chatHistory.scrollTop = topBefore + (chatHistory.scrollHeight - heightBefore);
            chatHistory.style.scrollBehavior = '';
        } catch (error) {
            if (seq !== viewSeq) return;
            button.disabled = false;
            button.textContent = `Retry loading earlier messages (${error.message})`;
        }
    }

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

    async function openStream(conversationId, message, apiKey, signal) {
        const url = `${API_BASE}/stream/conversation/${encodeURIComponent(conversationId)}`;

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

    function buildMessage(role, content, steps, sources) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-msg`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'msg-content';
        contentDiv.textContent = content;
        msgDiv.appendChild(contentDiv);

        const sourcesEl = buildSources(sources);
        if (sourcesEl) contentDiv.appendChild(sourcesEl);

        if (steps && steps.length) {
            contentDiv.appendChild(buildTrace(steps));
        }
        return msgDiv;
    }

    function appendMessage(role, content, steps, sources) {
        const msgDiv = buildMessage(role, content, steps, sources);
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    /* Links the agents cited. They come off the web, so only http(s) becomes a link. */
    function buildSources(sources) {
        const links = [];
        for (const raw of sources || []) {
            let url;
            try {
                url = new URL(raw);
            } catch {
                continue;
            }
            if (url.protocol !== 'http:' && url.protocol !== 'https:') continue;

            const a = document.createElement('a');
            a.href = url.href;
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
            a.title = url.href;
            a.textContent = url.hostname.replace(/^www\./, '') + (url.pathname === '/' ? '' : url.pathname);
            links.push(a);
        }
        if (!links.length) return null;

        const box = document.createElement('div');
        box.className = 'sources';

        const title = document.createElement('div');
        title.className = 'sources-title';
        title.textContent = `Sources (${links.length})`;

        const list = document.createElement('ol');
        list.className = 'sources-list';
        for (const a of links) {
            const li = document.createElement('li');
            li.appendChild(a);
            list.appendChild(li);
        }

        box.append(title, list);
        return box;
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

        // A fresh chat gets its id here; the backend takes any id in the path.
        if (!currentConversationId) currentConversationId = newConversationId();
        const conversationId = currentConversationId;

        chatInput.value = '';
        setBusy(true);
        appendMessage('user', message);

        const progressEl = appendProgress();
        const tracker = createTurnTracker(progressEl.querySelector('.progress-steps'));
        const controller = new AbortController();
        activeStream = controller;

        try {
            const response = await openStream(conversationId, message, apiKey, controller.signal);
            // The server took the turn: only now does it belong in the chat list.
            touchChat(conversationId, message);
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
                if (terminal.event === 'done') {
                    appendMessage(
                        'assistant',
                        terminal.data.conversation || 'No response content.',
                        tracker.steps,
                        terminal.data.sources,
                    );
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
            if (activeStream === controller) activeStream = null;
            setBusy(false);
        }
    });

    /* ---------------------------------------------------------------- boot */

    localStorage.removeItem('finagent_chats'); // the list used to be kept here; it now comes from the server
    renderChatList();
    renderHeader();
    loadConversations();
    if (currentConversationId) openChat(currentConversationId);
});
