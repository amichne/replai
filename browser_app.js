(function () {
  'use strict';

  let currentDoc = INITIAL_RAW_DOC;
  let currentEvents = Array.isArray(INITIAL_EVENTS) ? INITIAL_EVENTS.slice() : [];
  let currentError = '';
  let stepMap = new Map();
  let stepOrder = [];
  let stepDocMap = new Map();
  let collapsedSteps = new Set();
  let showReasoning = false;
  let showMetrics = true;
  let filterText = '';

  const ROLE_LABEL = { user: 'USER', agent: 'AGENT', system: 'SYSTEM', tool: 'TOOL' };
  const ROLE_CLASS = { user: 'r-user', agent: 'r-agent', system: 'r-system', tool: 'r-tool' };
  const ATTACHMENTS_BLOCK_RE = /<attachments>[\s\S]*?<\/attachments>/gi;
  const ATTACHMENT_RE = /<attachment\b[^>]*>[\s\S]*?<\/attachment>/gi;
  const USER_REQUEST_RE = /<userRequest>\s*([\s\S]*?)\s*<\/userRequest>/gi;
  const PREVIEW_LIMIT = 220;

  function stringify(value) {
    if (typeof value === 'string') return value;
    try {
      return JSON.stringify(value, null, 2);
    } catch (_err) {
      return String(value);
    }
  }

  function summarizeText(value, limit) {
    const collapsed = stringify(value).replace(/\s+/g, ' ').trim();
    return collapsed.length > limit ? collapsed.slice(0, limit - 1) + '…' : collapsed;
  }

  function firstFiniteNumber() {
    for (let i = 0; i < arguments.length; i += 1) {
      const value = Number(arguments[i]);
      if (Number.isFinite(value)) return value;
    }
    return null;
  }

  function parseTimestampMs(ts) {
    if (!ts) return null;
    const value = Date.parse(String(ts));
    return Number.isFinite(value) ? value : null;
  }

  function formatCount(value) {
    const num = Number(value || 0);
    return Number.isFinite(num) ? num.toLocaleString('en-US') : '0';
  }

  function formatCurrency(value) {
    const num = Number(value || 0);
    if (!Number.isFinite(num)) return '$0.0000';
    return '$' + num.toFixed(4);
  }

  function formatDuration(ms) {
    const value = Number(ms || 0);
    if (!Number.isFinite(value) || value < 0) return 'n/a';
    if (value < 1000) return Math.round(value) + ' ms';
    if (value < 60000) return (value / 1000).toFixed(value >= 10000 ? 0 : 1) + ' s';
    const minutes = Math.floor(value / 60000);
    const seconds = Math.round((value % 60000) / 1000);
    return minutes + 'm ' + seconds + 's';
  }

  function normalizeMetricPayload(metrics) {
    const body = (metrics && typeof metrics === 'object' && !Array.isArray(metrics)) ? metrics : {};
    const usage = (body.usage && typeof body.usage === 'object' && !Array.isArray(body.usage)) ? body.usage : {};
    const tokenUsage = (body.token_usage && typeof body.token_usage === 'object' && !Array.isArray(body.token_usage)) ? body.token_usage : {};
    const prompt = firstFiniteNumber(
      body.prompt_tokens,
      body.prompt_token_count,
      body.input_tokens,
      usage.prompt_tokens,
      usage.prompt_token_count,
      usage.input_tokens,
      tokenUsage.prompt_tokens,
      tokenUsage.input_tokens
    ) || 0;
    const completion = firstFiniteNumber(
      body.completion_tokens,
      body.completion_token_count,
      body.output_tokens,
      usage.completion_tokens,
      usage.completion_token_count,
      usage.output_tokens,
      tokenUsage.completion_tokens,
      tokenUsage.output_tokens
    ) || 0;
    let total = firstFiniteNumber(
      body.total_tokens,
      body.total_token_count,
      usage.total_tokens,
      usage.total_token_count,
      tokenUsage.total_tokens
    );
    if (total == null && (prompt || completion)) total = prompt + completion;
    const cost = firstFiniteNumber(
      body.cost_usd,
      usage.cost_usd,
      tokenUsage.cost_usd,
      body.cost,
      usage.cost,
      tokenUsage.cost
    ) || 0;
    return { prompt: prompt, completion: completion, total: total || 0, cost: cost, raw: body };
  }

  function setError(message) {
    currentError = String(message || '');
    renderConversation();
  }

  function clearError() {
    currentError = '';
  }

  function decodeArgs(value) {
    if (typeof value !== 'string') return value;
    try {
      return JSON.parse(value);
    } catch (_err) {
      return value;
    }
  }

  function requireCondition(condition, message) {
    if (!condition) throw new Error(message);
  }

  function copilotToAtif(raw) {
    if (raw && typeof raw === 'object' && Array.isArray(raw.steps)) return raw;

    const messages = (raw && Array.isArray(raw.messages)) ? raw.messages : [];
    const roleMap = { assistant: 'agent', user: 'user', system: 'system' };
    const steps = messages.map(function (message, i) {
      const rawRole = (message && message.role) || 'user';
      const role = roleMap[rawRole] || 'user';
      const step = {
        message: {
          role: role,
          content: (message && message.content) || ''
        }
      };
      const rawToolCalls = (message && Array.isArray(message.tool_calls)) ? message.tool_calls : [];
      if (rawToolCalls.length) {
        step.tool_calls = rawToolCalls.map(function (tc, j) {
          const fn = (tc && tc.function) || {};
          return {
            id: (tc && tc.id) || ('tc_' + i + '_' + j),
            name: fn.name || 'unknown',
            arguments: decodeArgs(fn.arguments || '{}')
          };
        });
      }
      return step;
    });

    return {
      schema_version: '1.5',
      session_id: (raw && raw.id) || 'converted',
      agent: { name: (raw && raw.model) || 'unknown' },
      steps: steps
    };
  }

  function validateDoc(doc) {
    requireCondition(doc && typeof doc === 'object' && !Array.isArray(doc), 'Root must be a JSON object.');
    requireCondition(typeof doc.schema_version === 'string' && /^(?:ATIF-v)?1\.5(?:\..*)?$/.test(doc.schema_version), 'Expected schema_version matching (ATIF-v)1.5.*.');
    requireCondition(Object.prototype.hasOwnProperty.call(doc, 'session_id'), 'Missing required field: session_id.');
    requireCondition(doc.agent && typeof doc.agent === 'object' && !Array.isArray(doc.agent), 'Missing or invalid agent object.');
    requireCondition(Array.isArray(doc.steps), 'Missing or invalid steps list.');

    doc.steps.forEach(function (step, i) {
      requireCondition(step && typeof step === 'object' && !Array.isArray(step), 'Step ' + i + ': must be a JSON object.');
      const msg = step.message;
      if (msg != null) {
        requireCondition(typeof msg === 'string' || (msg && typeof msg === 'object' && !Array.isArray(msg)), 'Step ' + i + ': message must be an object or string.');
        if (msg && typeof msg === 'object' && !Array.isArray(msg)) {
          requireCondition(Object.prototype.hasOwnProperty.call(msg, 'content'), 'Step ' + i + ': message object missing content.');
          requireCondition(Object.prototype.hasOwnProperty.call(msg, 'role') || Object.prototype.hasOwnProperty.call(step, 'source'), 'Step ' + i + ': message object missing role and step missing source.');
        }
      }
      if (step.source != null) {
        requireCondition(typeof step.source === 'string', 'Step ' + i + ': source must be a string.');
      }
      if (step.tool_calls != null) {
        requireCondition(Array.isArray(step.tool_calls), 'Step ' + i + ': tool_calls must be a list.');
        step.tool_calls.forEach(function (tc, j) {
          requireCondition(tc && typeof tc === 'object' && !Array.isArray(tc), 'Step ' + i + ', tool_call ' + j + ': must be an object.');
          requireCondition(Object.prototype.hasOwnProperty.call(tc, 'id') || Object.prototype.hasOwnProperty.call(tc, 'tool_call_id'), 'Step ' + i + ', tool_call ' + j + ': missing id/tool_call_id.');
          requireCondition(Object.prototype.hasOwnProperty.call(tc, 'name') || Object.prototype.hasOwnProperty.call(tc, 'function_name'), 'Step ' + i + ', tool_call ' + j + ': missing name/function_name.');
        });
      }
      if (step.observation != null) {
        requireCondition(step.observation && typeof step.observation === 'object' && !Array.isArray(step.observation), 'Step ' + i + ': observation must be an object.');
        if (step.observation.results != null) {
          requireCondition(Array.isArray(step.observation.results), 'Step ' + i + ': observation.results must be a list.');
          step.observation.results.forEach(function (result, k) {
            requireCondition(result && typeof result === 'object' && !Array.isArray(result), 'Step ' + i + ', result ' + k + ': must be an object.');
            requireCondition(Object.prototype.hasOwnProperty.call(result, 'source_call_id') || Object.prototype.hasOwnProperty.call(result, 'ref'), 'Step ' + i + ', result ' + k + ': missing source_call_id/ref.');
          });
        }
      }
    });
  }

  function canonicalRole(raw) {
    const table = {
      user: 'user',
      human: 'user',
      assistant: 'agent',
      agent: 'agent',
      system: 'system',
      tool: 'tool',
      function: 'tool'
    };
    return table[String(raw || 'user').toLowerCase()] || 'user';
  }

  function stripAttachmentContent(value) {
    if (typeof value !== 'string') return value;
    let text = value;
    let hadAttachments = false;
    const beforeA = text;
    text = text.replace(ATTACHMENTS_BLOCK_RE, '');
    hadAttachments = hadAttachments || text !== beforeA;
    const beforeB = text;
    text = text.replace(ATTACHMENT_RE, '');
    hadAttachments = hadAttachments || text !== beforeB;
    text = text.replace(/\n{3,}/g, '\n\n').trim();
    if (hadAttachments && !text) return '[attachments omitted]';
    return text;
  }

  function extractUserRequestContent(value) {
    if (typeof value !== 'string') return value;
    const matches = [];
    let match;
    USER_REQUEST_RE.lastIndex = 0;
    while ((match = USER_REQUEST_RE.exec(value)) !== null) {
      if (match[1] && match[1].trim()) matches.push(match[1].trim());
    }
    return matches.length ? matches.join('\n\n') : value;
  }

  function normalizeMessageBody(value, role) {
    let body = stripAttachmentContent(value);
    if (role === 'user') body = extractUserRequestContent(body);
    return body;
  }

  function parseToolCall(raw) {
    return {
      call_id: raw.id || raw.tool_call_id || 'unknown',
      name: raw.name || raw.function_name || 'unknown',
      arguments: decodeArgs(raw.arguments == null ? {} : raw.arguments),
      timestamp: raw.timestamp || null
    };
  }

  function parseToolResult(raw) {
    return {
      source_call_id: raw.source_call_id || raw.ref || null,
      content: raw.content,
      timestamp: raw.timestamp || null
    };
  }

  function normalizeDoc(doc) {
    const events = [];

    (doc.steps || []).forEach(function (step, stepIndex) {
      const stepId = Number(step.step_id || (stepIndex + 1));
      const rawMessage = step.message;
      let stepTs = step.timestamp || null;

      if (rawMessage != null) {
        let role;
        let body;
        if (rawMessage && typeof rawMessage === 'object' && !Array.isArray(rawMessage)) {
          stepTs = rawMessage.timestamp || stepTs;
          role = canonicalRole(rawMessage.role || step.source || 'user');
          body = normalizeMessageBody(rawMessage.content || '', role);
        } else {
          role = canonicalRole(String(step.source || 'user'));
          body = normalizeMessageBody(String(rawMessage), role);
        }
        events.push({ kind: 'message', step_id: stepId, ts: stepTs, role: role, title: null, body: body, ref: null });
      }

      if (step.reasoning_content != null) {
        events.push({ kind: 'reasoning', step_id: stepId, ts: stepTs, role: 'agent', title: 'reasoning', body: step.reasoning_content, ref: null });
      }

      (step.tool_calls || []).forEach(function (rawTc) {
        const tc = parseToolCall(rawTc);
        events.push({ kind: 'tool_call', step_id: stepId, ts: tc.timestamp || stepTs, role: 'agent', title: tc.name, body: tc.arguments, ref: tc.call_id });
      });

      const observation = step.observation || {};
      (observation.results || []).forEach(function (rawResult) {
        const result = parseToolResult(rawResult);
        events.push({ kind: 'tool_result', step_id: stepId, ts: result.timestamp || stepTs, role: 'tool', title: null, body: result.content, ref: result.source_call_id });
      });

      if (step.metrics != null) {
        events.push({ kind: 'metrics', step_id: stepId, ts: null, role: 'agent', title: 'metrics', body: step.metrics, ref: null });
      }
    });

    if (doc.final_metrics != null) {
      const finalStepId = events.length ? (events[events.length - 1].step_id + 1) : 1;
      events.push({ kind: 'metrics', step_id: finalStepId, ts: null, role: 'agent', title: 'final_metrics', body: doc.final_metrics, ref: null });
    }

    return events;
  }

  function hashStepId() {
    const match = window.location.hash.match(/^#step-(\d+)$/);
    return match ? parseInt(match[1], 10) : null;
  }

  function hasLoadedDoc() {
    return !!currentDoc;
  }

  function rebuildIndex() {
    stepMap = new Map();
    stepOrder = [];
    stepDocMap = new Map();

    currentEvents.forEach(function (ev) {
      if (!stepMap.has(ev.step_id)) {
        stepMap.set(ev.step_id, []);
        stepOrder.push(ev.step_id);
      }
      stepMap.get(ev.step_id).push(ev);
    });

    const steps = (currentDoc && Array.isArray(currentDoc.steps)) ? currentDoc.steps : [];
    steps.forEach(function (step, index) {
      const stepId = Number(step.step_id || (index + 1));
      if (!stepDocMap.has(stepId)) stepDocMap.set(stepId, step);
      if (!stepMap.has(stepId)) {
        stepMap.set(stepId, []);
        stepOrder.push(stepId);
      }
    });
  }

  function updateMeta() {
    const meta = document.getElementById('session-meta');
    const rawTitle = document.getElementById('raw-pane-title');
    const emptyState = slot(meta, 'empty');
    const loadedState = slot(meta, 'loaded');
    if (!hasLoadedDoc()) {
      if (emptyState) emptyState.hidden = false;
      if (loadedState) loadedState.hidden = true;
      rawTitle.textContent = 'Raw ATIF JSON';
      setRawButtonEnabled(false);
      return;
    }

    if (emptyState) emptyState.hidden = true;
    if (loadedState) loadedState.hidden = false;
    setSlotText(meta, 'session-id', String(currentDoc.session_id || 'unknown'));
    setSlotText(meta, 'agent-name', String((currentDoc.agent || {}).name || 'unknown'));
    setSlotText(meta, 'step-count', String(stepOrder.length) + ' steps');
    rawTitle.textContent = 'Raw ATIF JSON — ' + String(currentDoc.session_id || 'unknown');
    setRawButtonEnabled(true);
  }

  function syncToggleButtons() {
    const reasoningButton = document.getElementById('btn-reasoning');
    reasoningButton.classList.toggle('on', showReasoning);
    reasoningButton.textContent = showReasoning ? 'Hide Reasoning' : 'Show Reasoning';

    const metricsButton = document.getElementById('btn-metrics');
    metricsButton.classList.toggle('on', showMetrics);
    metricsButton.textContent = showMetrics ? 'Hide Metrics' : 'Show Metrics';
  }

  function setRawButtonEnabled(enabled) {
    const button = document.getElementById('btn-raw');
    button.disabled = !enabled;
    button.classList.toggle('disabled', !enabled);
  }

  function codeLanguageClass(value) {
    return value != null && typeof value === 'object' ? 'language-json' : 'language-plaintext';
  }

  function cloneTemplate(id) {
    const template = document.getElementById(id);
    if (!template || !template.content || !template.content.firstElementChild) {
      throw new Error('Missing template: ' + id);
    }
    return template.content.firstElementChild.cloneNode(true);
  }

  function slot(root, name) {
    return root.querySelector('[data-slot="' + name + '"]');
  }

  function setSlotText(root, name, value) {
    const node = slot(root, name);
    if (node) node.textContent = value == null ? '' : String(value);
    return node;
  }

  function clearNode(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function createCodeElement(value, languageClass) {
    const code = document.createElement('code');
    code.className = 'code-syntax ' + (languageClass || 'language-plaintext');
    code.textContent = String(value == null ? '' : value);
    return code;
  }

  function createCodeBlock(value, languageClass) {
    const pre = document.createElement('pre');
    pre.className = 'code-block';
    pre.appendChild(createCodeElement(value, languageClass));
    return pre;
  }

  function appendNodes(parent, nodes) {
    nodes.forEach(function (node) {
      if (node) parent.appendChild(node);
    });
  }

  function highlightCodeBlocks() {
    if (!window.hljs || typeof window.hljs.highlightElement !== 'function') return;
    document.querySelectorAll('code.code-syntax').forEach(function (block) {
      if (block.dataset.highlighted === 'true') return;
      window.hljs.highlightElement(block);
      block.dataset.highlighted = 'true';
    });
  }

  function scheduleHighlighting() {
    window.requestAnimationFrame(function () {
      highlightCodeBlocks();
    });
  }

  function getVisibleStepIds() {
    const f = filterText.trim().toLowerCase();
    if (!f) return stepOrder.slice();

    return stepOrder.filter(function (sid) {
      const events = stepMap.get(sid) || [];
      const preview = stepPreview(events).toLowerCase();
      const role = primaryRole(events).toLowerCase();
      const payload = events.map(function (ev) {
        return [ev.kind, ev.title || '', summarizeText(ev.body, 400)].join(' ');
      }).join(' ').toLowerCase();
      return String(sid).includes(f) || role.includes(f) || preview.includes(f) || payload.includes(f);
    });
  }

  function setCollapseAllButtonLabel() {
    const button = document.getElementById('btn-collapse-all');
    const visibleIds = getVisibleStepIds();
    const anyExpanded = visibleIds.some(function (stepId) { return !collapsedSteps.has(stepId); });
    button.textContent = anyExpanded ? 'Fold All' : 'Expand All';
    button.disabled = !visibleIds.length;
    button.classList.toggle('disabled', !visibleIds.length);
  }

  function primaryRole(events) {
    for (const ev of events) {
      if (ev.kind === 'message') return ev.role;
    }
    return (events[0] && events[0].role) || 'agent';
  }

  function stepPreview(events) {
    for (const ev of events) {
      if (ev.kind === 'message' && ev.body != null) {
        return summarizeText(ev.body, 72);
      }
    }
    for (const ev of events) {
      if (ev.kind === 'tool_call') return 'tool: ' + (ev.title || '?') + '(…)';
    }
    return '';
  }

  function primaryTs(events) {
    for (const ev of events) {
      if (ev.ts) return ev.ts.replace('T', ' ').replace('Z', '').slice(0, 19);
    }
    return null;
  }

  function buildEmptyState() {
    const node = cloneTemplate('tpl-empty-state');
    const title = hasLoadedDoc() ? 'Nothing to show' : 'Open a trajectory';
    const message = hasLoadedDoc()
      ? 'No steps match the current filter.'
      : 'Open a local ATIF or Copilot-style JSON trajectory to explore it in a single scrollable chain.';
    setSlotText(node, 'title', title);
    setSlotText(node, 'message', message);
    slot(node, 'extra').hidden = hasLoadedDoc();
    return node;
  }

  function createInlineError(message) {
    const node = cloneTemplate('tpl-inline-error');
    setSlotText(node, 'message', message);
    return node;
  }

  function createJsonDetailNodes(summary, value) {
    const full = stringify(value);
    const needsExpand = full.length > PREVIEW_LIMIT || /\n/.test(full) || typeof value !== 'string';
    const nodes = [];
    if (!full) return nodes;

    const snippet = document.createElement('div');
    snippet.className = 'event-snippet';
    snippet.textContent = "";
    nodes.push(snippet);

    if (!needsExpand) return nodes;

    const detail = cloneTemplate('tpl-json-detail');
    setSlotText(detail, 'summary', summary);
    slot(detail, 'content').appendChild(createCodeBlock(full, codeLanguageClass(value)));
    nodes.push(detail);
    return nodes;
  }

  function renderMessageEvent(ev) {
    const node = cloneTemplate('tpl-event-message');
    const body = slot(node, 'body');
    body.classList.add(ev.role || 'agent');
    body.textContent = stringify(ev.body);
    return node;
  }

  function renderReasoningEvent(ev) {
    if (!showReasoning) return null;
    const node = cloneTemplate('tpl-event-reasoning');
    setSlotText(node, 'body', stringify(ev.body));
    return node;
  }

  function renderToolCallEvent(ev) {
    const node = cloneTemplate('tpl-event-tool-call');
    const ref = slot(node, 'ref');
    if (ev.ref) {
      ref.hidden = false;
      ref.textContent = '#' + ev.ref;
    }
    setSlotText(node, 'title', ev.title || '?');
    appendNodes(node, createJsonDetailNodes('Expand tool arguments JSON', ev.body));
    return node;
  }

  function renderToolResultEvent(ev) {
    const node = cloneTemplate('tpl-event-tool-result');
    const ref = slot(node, 'ref');
    if (ev.ref) {
      ref.hidden = false;
      ref.textContent = '#' + ev.ref;
    }
    appendNodes(node, createJsonDetailNodes('Expand full result JSON', ev.body));
    return node;
  }

  function renderMetricsEvent(ev) {
    if (!showMetrics) return null;
    const metrics = normalizeMetricPayload(ev.body).raw;
    const node = cloneTemplate('tpl-event-metrics');
    const detail = cloneTemplate('tpl-json-detail');
    setSlotText(detail, 'summary', 'Expand metrics JSON');
    slot(detail, 'content').appendChild(createCodeBlock(stringify(metrics), 'language-json'));
    node.appendChild(detail);
    return node;
  }

  function computeAnalytics(stepIds) {
    const ids = Array.isArray(stepIds) ? stepIds.slice() : stepOrder.slice();
    const summary = {
      steps: ids.length,
      messages: 0,
      reasoning: 0,
      toolCalls: 0,
      toolResults: 0,
      metricEvents: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
      costUsd: 0
    };
    const toolCounts = new Map();
    const perStep = [];

    ids.forEach(function (stepId) {
      const events = stepMap.get(stepId) || [];
      const stepInfo = {
        stepId: stepId,
        role: primaryRole(events),
        preview: stepPreview(events),
        startMs: null,
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        costUsd: 0,
        toolCalls: 0,
        toolNames: []
      };

      events.forEach(function (ev) {
        if (stepInfo.startMs == null) stepInfo.startMs = parseTimestampMs(ev.ts);
        if (ev.kind === 'message') summary.messages += 1;
        if (ev.kind === 'reasoning') summary.reasoning += 1;
        if (ev.kind === 'tool_call') {
          const toolName = String(ev.title || '?');
          summary.toolCalls += 1;
          stepInfo.toolCalls += 1;
          stepInfo.toolNames.push(toolName);
          toolCounts.set(toolName, (toolCounts.get(toolName) || 0) + 1);
        }
        if (ev.kind === 'tool_result') summary.toolResults += 1;
        if (ev.kind === 'metrics') {
          const stepMetrics = normalizeMetricPayload(ev.body);
          summary.metricEvents += 1;
          summary.promptTokens += stepMetrics.prompt;
          summary.completionTokens += stepMetrics.completion;
          summary.totalTokens += stepMetrics.total;
          summary.costUsd += stepMetrics.cost;
          stepInfo.promptTokens += stepMetrics.prompt;
          stepInfo.completionTokens += stepMetrics.completion;
          stepInfo.totalTokens += stepMetrics.total;
          stepInfo.costUsd += stepMetrics.cost;
        }
      });

      perStep.push(stepInfo);
    });

    if (!summary.totalTokens && (summary.promptTokens || summary.completionTokens)) {
      summary.totalTokens = summary.promptTokens + summary.completionTokens;
    }

    let longestIteration = null;
    for (let i = 0; i < perStep.length - 1; i += 1) {
      const current = perStep[i];
      const next = perStep[i + 1];
      if (current.startMs == null || next.startMs == null) continue;
      const deltaMs = next.startMs - current.startMs;
      if (!Number.isFinite(deltaMs) || deltaMs < 0) continue;
      if (!longestIteration || deltaMs > longestIteration.durationMs) {
        longestIteration = {
          stepId: current.stepId,
          nextStepId: next.stepId,
          durationMs: deltaMs,
          preview: current.preview || ''
        };
      }
    }

    const topTools = Array.from(toolCounts.entries())
      .sort(function (a, b) {
        if (b[1] !== a[1]) return b[1] - a[1];
        return a[0].localeCompare(b[0]);
      })
      .map(function (entry) {
        return { name: entry[0], count: entry[1] };
      });

    const tokenSteps = perStep.filter(function (step) {
      return step.totalTokens || step.promptTokens || step.completionTokens || step.costUsd;
    });
    const maxTokenTotal = tokenSteps.reduce(function (maxValue, step) {
      const total = step.totalTokens || (step.promptTokens + step.completionTokens);
      return Math.max(maxValue, total || 0);
    }, 0);

    return {
      summary: summary,
      perStep: perStep,
      topTools: topTools,
      tokenSteps: tokenSteps,
      maxTokenTotal: maxTokenTotal,
      longestIteration: longestIteration
    };
  }

  function createSidebarPlaceholder(eyebrow, title, copy) {
    const node = cloneTemplate('tpl-sidebar-placeholder');
    setSlotText(node, 'eyebrow', eyebrow);
    setSlotText(node, 'title', title);
    const copyNode = setSlotText(node, 'copy', copy || '');
    if (copyNode) copyNode.hidden = !copy;
    return node;
  }

  function createAnalyticsSummaryRow(label, value, meta) {
    const node = cloneTemplate('tpl-analytics-summary-row');
    setSlotText(node, 'label', label);
    setSlotText(node, 'value', value);
    const metaNode = setSlotText(node, 'meta', meta || '');
    if (metaNode) metaNode.hidden = !meta;
    return node;
  }

  function createAnalyticsRow(primary, secondary, value, barWidthPercent) {
    const node = cloneTemplate('tpl-analytics-row');
    setSlotText(node, 'primary', primary);
    const secondaryNode = setSlotText(node, 'secondary', secondary || '');
    if (secondaryNode) secondaryNode.hidden = !secondary;
    setSlotText(node, 'value', value);
    const barNode = slot(node, 'bar');
    if (barNode && Number.isFinite(barWidthPercent) && barWidthPercent > 0) {
      barNode.hidden = false;
      slot(node, 'bar-fill').style.width = Math.max(0, Math.min(100, barWidthPercent)) + '%';
    }
    return node;
  }

  function createAnalyticsEmpty(message) {
    const node = cloneTemplate('tpl-analytics-empty');
    setSlotText(node, 'message', message);
    return node;
  }

  function createAnalyticsSection(title, children) {
    const node = cloneTemplate('tpl-analytics-section');
    setSlotText(node, 'title', title);
    appendNodes(slot(node, 'list'), children);
    return node;
  }

  function buildAnalyticsPanel(stepIds) {
    const analytics = computeAnalytics(stepIds);
    const filtered = stepIds.length !== stepOrder.length;
    const subtitle = filtered
      ? 'Filtered view: ' + formatCount(stepIds.length) + ' of ' + formatCount(stepOrder.length) + ' steps.'
      : 'Whole conversation.';
    const longestText = analytics.longestIteration
      ? 'step ' + analytics.longestIteration.stepId + ' → ' + analytics.longestIteration.nextStepId + ' · ' + formatDuration(analytics.longestIteration.durationMs)
      : 'n/a';

    const node = cloneTemplate('tpl-analytics-panel');
    setSlotText(node, 'subtitle', subtitle);

    appendNodes(slot(node, 'summary'), [
      createAnalyticsSummaryRow('Steps', formatCount(analytics.summary.steps), formatCount(analytics.summary.messages) + ' messages · ' + formatCount(analytics.summary.reasoning) + ' reasoning blocks'),
      createAnalyticsSummaryRow('Tool calls', formatCount(analytics.summary.toolCalls), formatCount(analytics.summary.toolResults) + ' tool results'),
      createAnalyticsSummaryRow('Tokens', formatCount(analytics.summary.totalTokens || (analytics.summary.promptTokens + analytics.summary.completionTokens)), 'prompt ' + formatCount(analytics.summary.promptTokens) + ' · completion ' + formatCount(analytics.summary.completionTokens)),
      createAnalyticsSummaryRow('Spend', formatCurrency(analytics.summary.costUsd), formatCount(analytics.summary.metricEvents) + ' metric events'),
      createAnalyticsSummaryRow('Longest iteration', longestText, analytics.longestIteration && analytics.longestIteration.preview ? analytics.longestIteration.preview : 'Measured from step timestamps')
    ]);

    const toolItems = analytics.topTools.length
      ? analytics.topTools.slice(0, 8).map(function (tool) {
          return createAnalyticsRow(tool.name, 'tool calls', formatCount(tool.count));
        })
      : [createAnalyticsEmpty('No tool calls in the current view.')];

    const tokenItems = analytics.tokenSteps.length
      ? analytics.tokenSteps.map(function (step) {
          const total = step.totalTokens || (step.promptTokens + step.completionTokens);
          const width = analytics.maxTokenTotal > 0 ? Math.max(4, Math.round((total / analytics.maxTokenTotal) * 100)) : 0;
          const preview = step.preview || ROLE_LABEL[step.role] || 'step';
          return createAnalyticsRow('[' + step.stepId + ']', preview, formatCount(total), width);
        })
      : [createAnalyticsEmpty('No token or cost metrics were found in the current view.')];

    appendNodes(slot(node, 'sections'), [
      createAnalyticsSection('Top tools', toolItems),
      createAnalyticsSection('Token usage by step', tokenItems)
    ]);

    return node;
  }

  function renderSidebar(stepIds) {
    const app = document.getElementById('app');
    const sidebar = document.getElementById('sidebar-content');
    app.classList.toggle('metrics-hidden', !showMetrics);

    if (!showMetrics) {
      sidebar.replaceChildren();
      return;
    }

    if (!hasLoadedDoc()) {
      sidebar.replaceChildren(createSidebarPlaceholder(
        'Analytics',
        'Metrics sidebar',
        'Load a trajectory to inspect token usage, tool activity, spend, and iteration timing alongside the conversation chain.'
      ));
      return;
    }

    if (!stepIds.length) {
      sidebar.replaceChildren(createSidebarPlaceholder(
        'Analytics',
        'No visible steps',
        'Your current filter hides every step. Clear or relax the filter to bring metrics back into view.'
      ));
      return;
    }

    sidebar.replaceChildren(buildAnalyticsPanel(stepIds));
  }

  function renderEvent(ev) {
    if (ev.kind === 'message') return renderMessageEvent(ev);
    if (ev.kind === 'reasoning') return renderReasoningEvent(ev);
    if (ev.kind === 'tool_call') return renderToolCallEvent(ev);
    if (ev.kind === 'tool_result') return renderToolResultEvent(ev);
    if (ev.kind === 'metrics') return renderMetricsEvent(ev);
    return null;
  }

  function createEmptyEvent() {
    return cloneTemplate('tpl-event-empty');
  }

  function createNodeJsonDetail(stepJson) {
    const node = cloneTemplate('tpl-node-json');
    setSlotText(node, 'summary', 'Full step JSON');
    slot(node, 'content').appendChild(createCodeBlock(stepJson, 'language-json'));
    return node;
  }

  function buildStepNode(stepId, events, isTarget) {
    const node = cloneTemplate('tpl-step-node');
    const role = primaryRole(events);
    const roleClass = ROLE_CLASS[role] || 'r-agent';
    const roleLabel = ROLE_LABEL[role] || role.toUpperCase();
    const preview = stepPreview(events);
    const ts = primaryTs(events);
    const collapsed = collapsedSteps.has(stepId);
    const stepDoc = stepDocMap.get(stepId);
    const stepJson = stepDoc != null ? stringify(stepDoc) : stringify(events);
    const renderedEvents = events.map(renderEvent).filter(Boolean);
    const roleNode = setSlotText(node, 'role', roleLabel);
    const tsNode = slot(node, 'timestamp');
    const toggleButton = slot(node, 'toggle');
    const linkButton = slot(node, 'link');
    const body = slot(node, 'body');

    node.id = 'step-' + stepId;
    node.dataset.stepId = String(stepId);
    node.classList.add('role-' + role);
    if (collapsed) node.classList.add('collapsed');
    if (isTarget) node.classList.add('is-target');

    setSlotText(node, 'step', '[' + stepId + ']');
    roleNode.classList.add(roleClass);
    setSlotText(node, 'preview', preview || '(no preview)');
    if (ts) {
      tsNode.hidden = false;
      tsNode.textContent = ts;
    } else {
      tsNode.hidden = true;
      tsNode.textContent = '';
    }

    toggleButton.textContent = collapsed ? 'Expand' : 'Fold';
    toggleButton.setAttribute('data-step-id', String(stepId));
    linkButton.setAttribute('data-step-id', String(stepId));

    if (renderedEvents.length) {
      appendNodes(body, renderedEvents);
    } else {
      body.appendChild(createEmptyEvent());
    }
    body.appendChild(createNodeJsonDetail(stepJson));
    return node;
  }

  function renderConversation() {
    const container = document.getElementById('conversation');
    const chainInner = document.createElement('div');
    chainInner.className = 'chain-inner';
    if (currentError) {
      chainInner.appendChild(createInlineError(currentError));
    }
    const visibleStepIds = getVisibleStepIds();
    renderSidebar(visibleStepIds);
    if (!visibleStepIds.length) {
      chainInner.appendChild(buildEmptyState());
      container.replaceChildren(chainInner);
      syncToggleButtons();
      setCollapseAllButtonLabel();
      scheduleHighlighting();
      return;
    }
    const targetStepId = hashStepId();
    visibleStepIds.forEach(function (stepId) {
      chainInner.appendChild(buildStepNode(stepId, stepMap.get(stepId) || [], stepId === targetStepId));
    });
    container.replaceChildren(chainInner);
    syncToggleButtons();
    setCollapseAllButtonLabel();
    scheduleHighlighting();
  }

  function applyDoc(doc) {
    currentDoc = doc;
    currentEvents = normalizeDoc(doc);
    collapsedSteps = new Set();
    clearError();
    rebuildIndex();
    updateMeta();
    renderConversation();
  }

  function loadFromRawObject(raw, fileLabel) {
    const doc = copilotToAtif(raw);
    validateDoc(doc);
    if (fileLabel && !doc.session_id) doc.session_id = fileLabel;
    applyDoc(doc);
  }

  function openLocalFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function () {
      try {
        const raw = JSON.parse(String(reader.result || ''));
        loadFromRawObject(raw, file.name);
        if (stepOrder.length) {
          navigateToStep(stepOrder[0], false);
        }
      } catch (err) {
        setError((err && err.message) ? err.message : String(err));
      }
    };
    reader.onerror = function () {
      setError('Failed to read the selected file.');
    };
    reader.readAsText(file);
  }

  function flashStep(stepId) {
    const el = document.getElementById('step-' + stepId);
    if (!el) return;
    el.classList.remove('flash');
    void el.offsetWidth;
    el.classList.add('flash');
    window.setTimeout(function () {
      el.classList.remove('flash');
    }, 1500);
  }

  function focusHashTarget(smooth) {
    const stepId = hashStepId();
    if (stepId == null) return;

    if (filterText && getVisibleStepIds().indexOf(stepId) === -1) {
      filterText = '';
      document.getElementById('filter-input').value = '';
      renderConversation();
    }

    if (collapsedSteps.has(stepId)) {
      collapsedSteps.delete(stepId);
      renderConversation();
    }

    const el = document.getElementById('step-' + stepId);
    if (!el) return;
    el.scrollIntoView({ block: 'center', behavior: smooth ? 'smooth' : 'auto' });
    flashStep(stepId);
  }

  function navigateToStep(stepId, smooth) {
    if (stepId == null) return;
    const nextHash = '#step-' + stepId;
    if (window.location.hash === nextHash) {
      renderConversation();
      focusHashTarget(!!smooth);
      return;
    }
    window.location.hash = nextHash;
  }

  function copyStepLink(stepId) {
    const url = new URL(window.location.href);
    url.hash = 'step-' + stepId;
    navigateToStep(stepId, true);
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url.toString()).catch(function () {
        // Best effort: the hash has already been updated.
      });
    }
  }

  document.getElementById('btn-open').addEventListener('click', function () {
    document.getElementById('file-input').click();
  });

  document.getElementById('file-input').addEventListener('change', function () {
    if (this.files && this.files[0]) openLocalFile(this.files[0]);
    this.value = '';
  });

  document.getElementById('filter-input').addEventListener('input', function () {
    filterText = this.value;
    renderConversation();
  });

  document.getElementById('btn-collapse-all').addEventListener('click', function () {
    const visibleStepIds = getVisibleStepIds();
    const anyExpanded = visibleStepIds.some(function (stepId) { return !collapsedSteps.has(stepId); });
    if (anyExpanded) {
      visibleStepIds.forEach(function (stepId) { collapsedSteps.add(stepId); });
    } else {
      visibleStepIds.forEach(function (stepId) { collapsedSteps.delete(stepId); });
    }
    renderConversation();
    if (window.location.hash) focusHashTarget(false);
  });

  document.getElementById('btn-reasoning').addEventListener('click', function () {
    showReasoning = !showReasoning;
    renderConversation();
  });

  document.getElementById('btn-metrics').addEventListener('click', function () {
    showMetrics = !showMetrics;
    renderConversation();
  });

  document.getElementById('btn-raw').addEventListener('click', function () {
    if (!hasLoadedDoc()) return;
    const rawContent = document.getElementById('raw-content');
    clearNode(rawContent);
    rawContent.appendChild(createCodeElement(JSON.stringify(currentDoc, null, 2), 'language-json'));
    document.getElementById('raw-modal').classList.add('open');
    scheduleHighlighting();
  });

  document.getElementById('close-raw').addEventListener('click', function () {
    document.getElementById('raw-modal').classList.remove('open');
  });

  document.getElementById('raw-modal').addEventListener('click', function (event) {
    if (event.target === this) this.classList.remove('open');
  });

  document.getElementById('conversation').addEventListener('click', function (event) {
    const action = event.target.closest('[data-action]');
    if (!action) return;

    const actionName = action.getAttribute('data-action');
    if (actionName === 'open-file') {
      document.getElementById('file-input').click();
      return;
    }

    const stepId = parseInt(action.getAttribute('data-step-id') || '', 10);
    if (!Number.isFinite(stepId)) return;

    if (actionName === 'toggle-step') {
      if (collapsedSteps.has(stepId)) collapsedSteps.delete(stepId);
      else collapsedSteps.add(stepId);
      renderConversation();
      if (window.location.hash === '#step-' + stepId) focusHashTarget(false);
    } else if (actionName === 'link-step') {
      copyStepLink(stepId);
    }
  });

  ['dragenter', 'dragover'].forEach(function (name) {
    document.getElementById('conversation').addEventListener(name, function (event) {
      event.preventDefault();
      this.classList.add('dragging');
    });
  });

  ['dragleave', 'dragend', 'drop'].forEach(function (name) {
    document.getElementById('conversation').addEventListener(name, function (event) {
      event.preventDefault();
      this.classList.remove('dragging');
    });
  });

  document.getElementById('conversation').addEventListener('drop', function (event) {
    const files = event.dataTransfer && event.dataTransfer.files;
    if (files && files[0]) openLocalFile(files[0]);
  });

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
      document.getElementById('raw-modal').classList.remove('open');
    }
  });

  document.addEventListener('keydown', function (event) {
    if (document.getElementById('raw-modal').classList.contains('open')) return;
    if (document.activeElement === document.getElementById('filter-input')) return;
    const visibleStepIds = getVisibleStepIds();
    if (!visibleStepIds.length) return;

    const currentStepId = hashStepId();
    let idx = visibleStepIds.indexOf(currentStepId);
    if (idx === -1) idx = 0;

    if (event.key === 'ArrowDown' || event.key === 'j') {
      if (idx < visibleStepIds.length - 1) navigateToStep(visibleStepIds[idx + 1], true);
    } else if (event.key === 'ArrowUp' || event.key === 'k') {
      if (idx > 0) navigateToStep(visibleStepIds[idx - 1], true);
    } else if (event.key === 'Home') {
      navigateToStep(visibleStepIds[0], true);
    } else if (event.key === 'End') {
      navigateToStep(visibleStepIds[visibleStepIds.length - 1], true);
    } else if (event.key === 'f') {
      const stepId = visibleStepIds[idx] || visibleStepIds[0];
      if (collapsedSteps.has(stepId)) collapsedSteps.delete(stepId);
      else collapsedSteps.add(stepId);
      renderConversation();
      if (window.location.hash === '#step-' + stepId) focusHashTarget(false);
    } else if (event.key === 'l') {
      const stepId = visibleStepIds[idx] || visibleStepIds[0];
      copyStepLink(stepId);
    }
  });

  window.addEventListener('hashchange', function () {
    renderConversation();
    focusHashTarget(true);
  });

  if (hasLoadedDoc()) {
    rebuildIndex();
  }
  document.addEventListener('DOMContentLoaded', function () {
    scheduleHighlighting();
  });
  updateMeta();
  syncToggleButtons();
  renderConversation();
  if (stepOrder.length && hashStepId() == null) {
    navigateToStep(stepOrder[0], false);
  } else {
    focusHashTarget(false);
  }
}());
