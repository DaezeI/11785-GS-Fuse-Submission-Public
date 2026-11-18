'use strict';

const REMOTE = 'https://raw.githubusercontent.com/DaezeI/11785-GS-Fuse-Submission-Public/main/research_home/records.json';
const statusLabel = (value) => ({ completed: 'Completed', running: 'Running', queued: 'Queued', submitted: 'Submitted', planned: 'Planned', awaiting_evidence: 'Awaiting evidence', failed: 'Failed', canceled: 'Canceled' }[String(value || '').toLowerCase()] || text(value, 'Status not reported'));
const $ = (id) => document.getElementById(id);
const text = (value, fallback = 'Not reported.') => typeof value === 'string' && value.trim() ? value.trim() : fallback;
function node(tag, className, content) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (content !== undefined) el.textContent = content;
  return el;
}
function safeUrl(value) {
  try { const u = new URL(value, location.href); return ['http:', 'https:'].includes(u.protocol) ? u.href : null; } catch (_) { return null; }
}
function link(label, url, className) {
  const href = safeUrl(url);
  if (!href) return null;
  const a = node('a', className, label); a.href = href; a.target = '_blank'; a.rel = 'noopener noreferrer'; return a;
}
function date(value) {
  if (!value) return 'Date not reported';
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? text(value) : d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'America/New_York' });
}
function statusType(status) {
  const s = text(status, '').toLowerCase();
  if (/complet|finish|done/.test(s)) return 'completed';
  if (s === 'running' || s === 'active' || s === 'in progress') return 'active';
  return 'pending';
}
function appendField(parent, label, value) {
  const box = node('div', 'field'); box.append(node('h4', '', label), node('p', '', text(value))); parent.append(box);
}
function render(data, fallback) {
  const experiments = Array.isArray(data.experiments) ? data.experiments.filter(x => x && typeof x === 'object') : [];
  const completed = experiments.filter(e => statusType(e.status) === 'completed');
  const active = experiments.filter(e => ['running', 'queued'].includes(e.status));
  $('stage').textContent = text(data.stage, 'PROJECT STATUS');
  $('page-title').textContent = text(data.title, 'GS-Fuse research home');
  $('overview').textContent = text(data.overview, 'A shared view of the questions, evidence, and next steps behind GS-Fuse.');
  $('source-status').className = 'source-status ' + (fallback ? 'fallback' : 'live');
  $('source-status').textContent = (fallback ? 'Showing the bundled fallback record, which may be out of date. Last reported: ' : 'Latest public record · Last updated: ') + date(data.updated_at) + '.';
  $('footer-updated').textContent = 'Last reported ' + date(data.updated_at);
  (Array.isArray(data.links) ? data.links : []).forEach(item => { const a = link(text(item.label, 'Project link') + ' ↗', item.url, 'text-link'); if (a) $('links').append(a); });
  if (completed.length) completed.forEach(e => {
    const card = node('article', 'learning-card');
    card.append(node('p', 'card-kicker', text(e.kind, 'EXPERIMENT')), node('h3', '', text(e.title, 'Untitled experiment')), node('p', 'finding', text(e.finding, 'No verified finding reported.')));
    const a = node('a', 'inline-link', 'View the record ↗'); a.href = '#record-' + encodeURIComponent(text(e.id, '')); card.append(a); $('learned-list').append(card);
  }); else $('learned-list').append(node('p', 'empty', 'No completed findings have been reported yet.'));
  if (active.length) active.forEach(e => {
    const row = node('article', 'running-row'); const main = node('div'); main.append(node('h3', '', text(e.title, 'Untitled experiment')), node('p', '', text(e.question, 'Question not reported.'))); row.append(node('span', 'pulse ' + e.status, ''), main, node('span', 'status ' + statusType(e.status), statusLabel(e.status))); $('running-list').append(row);
  }); else $('running-list').append(node('p', 'empty', 'No running or queued jobs were reported in the latest update.'));
  const next = Array.isArray(data.next) ? data.next : [];
  if (next.length) next.forEach(item => { const card = node('article', 'next-item'); const body = node('div'); body.append(node('h3', '', text(item.title, 'Next step')), node('p', '', text(item.reason))); card.append(body); $('next-list').append(card); });
  else $('next-list').append(node('p', 'empty', 'No next steps have been reported yet.'));
  if (!experiments.length) $('records-list').append(node('p', 'empty', 'No experiment records are available.'));
  const names = new Map(experiments.map(e => [String(e.id), text(e.title, String(e.id))]));
  experiments.forEach(e => {
    const details = node('details', 'record'); details.id = 'record-' + encodeURIComponent(text(e.id, ''));
    const summary = node('summary'); const head = node('span', 'record-head');
    head.append(node('span', 'record-meta', text(e.kind, 'EXPERIMENT') + ' · ' + text(e.owner, 'Owner not reported')), node('span', 'record-title', text(e.title, 'Untitled experiment')), node('span', 'record-question', text(e.question, 'Question not reported.')));
    summary.append(head, node('span', 'status ' + statusType(e.status), statusLabel(e.status)), node('span', 'expand-icon', '+')); details.append(summary);
    const body = node('div', 'record-body'); const grid = node('div', 'field-grid');
    appendField(grid, 'Question', e.question); appendField(grid, 'Configuration', e.configuration); appendField(grid, statusType(e.status) === 'completed' ? 'Finding' : 'Finding / current state', e.finding); appendField(grid, 'Limitations', e.limitations); body.append(grid);
    if (e.figure && safeUrl(e.figure.url)) { const fig = node('figure'); const img = node('img'); img.src = safeUrl(e.figure.url); img.alt = text(e.figure.alt, 'Experiment figure'); img.loading = 'lazy'; fig.append(img, node('figcaption', '', img.alt)); body.append(fig); }
    const evidence = Array.isArray(e.evidence) ? e.evidence : []; const evidenceBox = node('div', 'evidence'); evidenceBox.append(node('h4', '', 'Evidence')); let count = 0;
    evidence.forEach(item => { const a = link(text(item.label, 'Evidence') + ' ↗', item.url, 'inline-link'); if (a) { evidenceBox.append(a); count++; } }); if (!count) evidenceBox.append(node('p', '', 'No evidence links reported.')); body.append(evidenceBox);
    const related = Array.isArray(e.related) ? e.related : [];
    if (related.length) { const box = node('div', 'related'); box.append(node('h4', '', 'Related records')); related.forEach(id => { if (names.has(String(id))) { const a = node('a', 'inline-link', names.get(String(id)) + ' ↗'); a.href = '#record-' + encodeURIComponent(String(id)); box.append(a); } }); body.append(box); }
    body.append(node('p', 'record-date', 'Last reported ' + date(e.updated_at))); details.append(body); $('records-list').append(details);
  });
  const source = link('View source record ↗', data.source_url || REMOTE, 'source-link'); if (source) $('source-status').append(' ', source);
}
async function fetchJson(url) {
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 5000);
  try { const response = await fetch(url, { cache: 'no-store', signal: controller.signal }); if (!response.ok) throw new Error('HTTP ' + response.status); const data = await response.json(); if (!data || typeof data !== 'object' || !Array.isArray(data.experiments)) throw new Error('Invalid record'); return data; } finally { clearTimeout(timer); }
}
async function init() {
  let data, fallback = new URLSearchParams(location.search).get('local') === '1';
  try { if (!fallback) data = await fetchJson(REMOTE + '?v=' + Math.floor(Date.now() / 60000)); } catch (_) { fallback = true; }
  if (fallback) { try { data = await fetchJson('records.json'); } catch (_) { $('source-status').className = 'source-status error'; $('source-status').textContent = 'The latest record and bundled fallback could not be loaded. Please try again later.'; $('overview').textContent = 'Research records are temporarily unavailable.'; return; } }
  render(data, fallback);
}
document.addEventListener('click', event => { const a = event.target.closest('a[href^="#record-"]'); if (!a) return; const target = document.getElementById(a.getAttribute('href').slice(1)); if (target) target.open = true; });
init();
