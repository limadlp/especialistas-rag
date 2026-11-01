const $ = id => document.getElementById(id);
let specialists = [];
let activeSpecialistId = localStorage.getItem('active_specialist') || 'nei-gong';
let activeConversationId = null;
let busy = false;
let availableModels = [];
const conversationKey = id => `specialist_${id}_active_conversation`;
const thinkingKey = id => `thinking_${id}`;

function updateThinkingControl() {
  const model = availableModels.find(item => item.id === $('model').value);
  const allowed = model?.reasoning_options || [];
  const canEnable = allowed.some(option => ['on', 'low', 'medium', 'high'].includes(option));
  const canDisable = allowed.includes('off');
  const control = $('thinking');
  control.disabled = !canEnable || !canDisable;
  control.checked = canEnable && (!canDisable || localStorage.getItem(thinkingKey(model.id)) === 'true');
  $('thinking-note').textContent = !model ? 'Inicie o LM Studio para detectar o suporte a thinking.' :
    !canEnable ? 'Este modelo não oferece ajuste de thinking.' :
    !canDisable ? 'Este modelo mantém thinking ativo.' :
    allowed.includes('on') ? 'O modelo alterna entre on e off.' :
    'Ao ativar, será usado um nível de thinking aceito pelo modelo.';
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Erro ${response.status}`);
  return data;
}

async function loadSpecialists(selected = activeSpecialistId) {
  const data = await fetchJSON('/api/specialists');
  specialists = data.specialists;
  const select = $('specialist');
  select.replaceChildren();
  for (const item of specialists) {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${item.name} · ${item.book_count} ${item.book_count === 1 ? 'livro' : 'livros'}`;
    select.append(option);
  }
  if (!specialists.some(x => x.id === selected)) selected = specialists[0]?.id;
  select.value = selected;
  return selected;
}

async function selectSpecialist(id) {
  if (busy) return;
  activeSpecialistId = id;
  localStorage.setItem('active_specialist', id);
  activeConversationId = localStorage.getItem(conversationKey(id));
  const specialist = specialists.find(x => x.id === id);
  $('specialist').value = id;
  $('specialist-name').textContent = specialist?.name.toUpperCase() || id;
  $('specialist-description').textContent = specialist?.description || 'Pergunte sobre os livros desta biblioteca.';
  if ($('upload-language')) $('upload-language').value = specialist?.ocr_language || 'por';
  $('messages').replaceChildren();
  showWelcome();
  $('upload-status').textContent = '';
  await Promise.all([loadBooks(), loadConversations()]);
  if (activeConversationId) await openConversation(activeConversationId);
}

function showWelcome() {
  const welcome = document.createElement('div');
  welcome.className = 'welcome';
  welcome.innerHTML = '<div class="welcome-icon">◉</div><h2>Por onde começamos?</h2><p>Você pode perguntar sobre um conceito, propor uma hipótese ou comparar perspectivas.</p>';
  $('messages').append(welcome);
}

async function loadModels() {
  const select = $('model');
  const prior = select.value;
  select.innerHTML = '<option>Carregando modelos…</option>';
  try {
    const data = await fetchJSON('/api/models');
    availableModels = data.models;
    select.replaceChildren();
    for (const model of data.models) {
      const option = document.createElement('option');
      option.value = model.id;
      option.textContent = model.label + (model.loaded ? ' · carregado' : '');
      select.append(option);
    }
    if (!data.models.length) select.innerHTML = '<option value="">Nenhum modelo de chat detectado</option>';
    else select.value = data.models.some(x => x.id === prior) ? prior : (data.models.find(x => x.loaded) || data.models[0]).id;
    updateThinkingControl();
  } catch (error) {
    availableModels = [];
    select.innerHTML = '<option value="">LM Studio indisponível</option>';
    updateThinkingControl();
    $('status').textContent = error.message;
  }
}

async function loadBooks() {
  const id = activeSpecialistId;
  try {
    const data = await fetchJSON(`/api/specialists/${encodeURIComponent(id)}/books`);
    if (id !== activeSpecialistId) return;
    $('count').textContent = data.books.length;
    const container = $('books');
    container.replaceChildren();
    for (const book of data.books) {
      const node = document.createElement('div');
      node.className = 'book';
      const title = document.createElement('strong');
      title.textContent = book.title;
      const meta = document.createElement('span');
      meta.textContent = book.format === 'epub' ? `${book.author} · EPUB · ${book.sections} seções` : `${book.author} · ${book.pages} páginas`;
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'delete-book';
      remove.textContent = 'Excluir da biblioteca';
      remove.setAttribute('aria-label', `Excluir ${book.title} da biblioteca`);
      remove.onclick = () => deleteBook(book);
      node.append(title, meta, remove);
      container.append(node);
    }
    $('status').textContent = data.books.length ? `${data.books.length} ${data.books.length === 1 ? 'obra pronta' : 'obras prontas'} para consulta` : 'Biblioteca vazia. Adicione um PDF.';
  } catch (error) { $('status').textContent = error.message; }
}

async function deleteBook(book) {
  if (!confirm(`Excluir “${book.title}” da biblioteca deste especialista? As conversas anteriores serão preservadas.`)) return;
  try {
    await fetchJSON(`/api/specialists/${encodeURIComponent(activeSpecialistId)}/books/${encodeURIComponent(book.id)}`, {method: 'DELETE'});
    await Promise.all([loadBooks(), loadSpecialists()]);
    $('upload-status').textContent = `“${book.title}” foi excluído da biblioteca.`;
  } catch (error) { $('upload-status').textContent = `Falha ao excluir: ${error.message}`; }
}

async function uploadBooks(files) {
  const id = activeSpecialistId;
  const language = $('upload-language')?.value || specialists.find(x => x.id === id)?.ocr_language || 'por';
  const password = $('pdf-password').value;
  $('pdf-password').value = '';
  const input = $('book-file');
  const failures = [];
  let completed = 0;
  input.disabled = true;
  const showStatus = message => { if (id === activeSpecialistId) $('upload-status').textContent = message; };
  try {
    for (const [index, file] of files.entries()) {
      const prefix = `Livro ${index + 1}/${files.length} · ${file.name}`;
      const isEpub = file.name.toLowerCase().endsWith('.epub');
      showStatus(`${prefix} · Enviando…`);
      try {
        const data = await fetchJSON(`/api/specialists/${encodeURIComponent(id)}/books`, {
          method: 'POST', headers: {'Content-Type': isEpub ? 'application/epub+zip' : 'application/pdf', 'X-Filename': encodeURIComponent(file.name), 'X-OCR-Language': language, 'X-PDF-Password': encodeURIComponent(password)}, body: file
        });
        while (true) {
          const job = await fetchJSON(`/api/jobs/${data.job_id}`);
          const unit = isEpub ? 'seções' : 'páginas';
          if (job.status === 'failed') throw new Error(job.error || 'Falha na indexação');
          if (job.status === 'done') { completed++; break; }
          showStatus(`${prefix} · ${job.stage} · ${job.current}/${job.total} ${unit}`);
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      } catch (error) {
        failures.push(`${file.name}: ${error.message}`);
      }
      try {
        await loadSpecialists(activeSpecialistId);
        if (id === activeSpecialistId) await loadBooks();
      } catch (error) {
        console.error('Falha ao atualizar a biblioteca:', error);
      }
    }
    showStatus(`${completed} de ${files.length} ${files.length === 1 ? 'livro indexado' : 'livros indexados'}.${failures.length ? ` Falhas: ${failures.join('; ')}` : ''}`);
  } finally {
    input.disabled = false;
  }
}

function addMessage(role, text, sources = [], retrieval = null, scroll = true) {
  $('messages').querySelector('.welcome')?.remove();
  const node = document.createElement('div');
  node.className = `message ${role}`;
  const label = document.createElement('div');
  label.className = 'role';
  label.textContent = role === 'user' ? 'VOCÊ' : `AGENTE ${specialists.find(x => x.id === activeSpecialistId)?.name.toUpperCase() || ''}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (role === 'assistant') bubble.append(renderAnswer(text, sources));
  else bubble.textContent = text;
  node.append(label, bubble);
  if (sources.length) {
    const list = document.createElement('div');
    list.className = 'sources';
    for (const source of sources) {
      const wrap = document.createElement('div');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'source';
      button.textContent = `[${source.citation}] ${source.author} · ${source.title} · ${sourceLocation(source)}`;
      const detail = document.createElement('div');
      detail.className = 'source-detail';
      detail.textContent = `[${source.citation}] ${source.id}\n${source.text}`;
      button.onclick = () => detail.classList.toggle('open');
      wrap.append(button, detail);
      list.append(wrap);
    }
    node.append(list);
  }
  if (retrieval?.search_rounds) {
    const note = document.createElement('div');
    note.className = 'retrieval-note';
    note.textContent = `Trechos reavaliados · ${retrieval.search_rounds} ${retrieval.search_rounds === 1 ? 'busca' : 'buscas'}`;
    node.append(note);
  }
  $('messages').append(node);
  if (scroll) node.scrollIntoView({block: 'end', behavior: 'smooth'});
  return node;
}

async function loadConversations() {
  const id = activeSpecialistId;
  const container = $('conversations');
  try {
    const data = await fetchJSON(`/api/specialists/${encodeURIComponent(id)}/conversations`);
    if (id !== activeSpecialistId) return;
    container.replaceChildren();
    if (!data.conversations.length) {
      const empty = document.createElement('p');
      empty.className = 'empty-conversations';
      empty.textContent = 'Suas conversas aparecerão aqui.';
      container.append(empty);
    }
    for (const conversation of data.conversations) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'conversation-item' + (conversation.id === activeConversationId ? ' active' : '');
      const title = document.createElement('strong');
      title.textContent = conversation.title;
      const date = document.createElement('span');
      date.textContent = new Date(conversation.updated_at).toLocaleString('pt-BR', {dateStyle: 'short', timeStyle: 'short'});
      button.append(title, date);
      button.onclick = () => openConversation(conversation.id);
      container.append(button);
    }
  } catch (error) { if (id === activeSpecialistId) container.textContent = `Não foi possível carregar as conversas: ${error.message}`; }
}

async function openConversation(id) {
  if (busy) return;
  const specialistId = activeSpecialistId;
  try {
    const conversation = await fetchJSON(`/api/specialists/${encodeURIComponent(specialistId)}/conversations/${encodeURIComponent(id)}`);
    if (specialistId !== activeSpecialistId) return;
    activeConversationId = id;
    localStorage.setItem(conversationKey(specialistId), id);
    $('messages').replaceChildren();
    for (const message of conversation.messages) addMessage(message.role, message.content, message.sources, message.retrieval, false);
    if (Array.from($('model').options).some(x => x.value === conversation.model)) $('model').value = conversation.model;
    await loadConversations();
    $('messages').scrollTop = $('messages').scrollHeight;
  } catch (error) { if (specialistId === activeSpecialistId) $('status').textContent = `Não foi possível abrir a conversa: ${error.message}`; }
}

async function send(question) {
  if (busy || !question.trim()) return;
  if (!$('model').value) { addMessage('assistant', 'Selecione um modelo do LM Studio antes de enviar.'); return; }
  busy = true;
  $('send').disabled = true;
  $('specialist').disabled = true;
  addMessage('user', question.trim());
  const pending = addMessage('assistant', 'Consultando os livros…');
  try {
    const data = await fetchJSON('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: question.trim(), model: $('model').value, specialist_id: activeSpecialistId,
        conversation_id: activeConversationId, thinking: $('thinking').checked})
    });
    pending.remove();
    addMessage('assistant', data.answer, data.sources, data.retrieval);
    activeConversationId = data.conversation_id;
    localStorage.setItem(conversationKey(activeSpecialistId), activeConversationId);
    await loadConversations();
  } catch (error) { pending.querySelector('.bubble').textContent = `Não foi possível responder: ${error.message}`; }
  finally { busy = false; $('send').disabled = false; $('specialist').disabled = false; $('question').focus(); }
}

$('composer').addEventListener('submit', event => { event.preventDefault(); const question = $('question').value; $('question').value = ''; send(question); });
$('question').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('composer').requestSubmit(); } });
$('refresh').onclick = loadModels;
$('model').onchange = updateThinkingControl;
$('thinking').onchange = () => localStorage.setItem(thinkingKey($('model').value), String($('thinking').checked));
$('edit-prompt').onclick = async () => {
  try {
    const data = await fetchJSON(`/api/specialists/${encodeURIComponent(activeSpecialistId)}/prompt`);
    $('system-prompt').value = data.system_prompt;
    $('prompt-error').textContent = '';
    $('prompt-dialog').showModal();
  } catch (error) { $('status').textContent = error.message; }
};
$('cancel-prompt').onclick = () => $('prompt-dialog').close();
$('prompt-form').onsubmit = async event => {
  event.preventDefault();
  try {
    await fetchJSON(`/api/specialists/${encodeURIComponent(activeSpecialistId)}/prompt`, {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({system_prompt: $('system-prompt').value})
    });
    $('prompt-dialog').close();
  } catch (error) { $('prompt-error').textContent = error.message; }
};
$('reset-prompt').onclick = async () => {
  try {
    const data = await fetchJSON(`/api/specialists/${encodeURIComponent(activeSpecialistId)}/prompt`, {
      method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({system_prompt: null})
    });
    $('system-prompt').value = data.system_prompt;
    $('prompt-error').textContent = '';
  } catch (error) { $('prompt-error').textContent = error.message; }
};
$('specialist').onchange = event => selectSpecialist(event.target.value);
$('new-chat').onclick = () => { if (busy) return; activeConversationId = null; localStorage.removeItem(conversationKey(activeSpecialistId)); $('messages').replaceChildren(); showWelcome(); loadConversations(); };
$('book-file').onchange = event => { const files = Array.from(event.target.files); event.target.value = ''; if (files.length) uploadBooks(files); };
$('create-specialist').onclick = () => $('specialist-dialog').showModal();
$('cancel-specialist').onclick = () => $('specialist-dialog').close();
$('specialist-form').onsubmit = async event => {
  event.preventDefault();
  const form = event.target;
  const values = Object.fromEntries(new FormData(form));
  try {
    const entry = await fetchJSON('/api/specialists', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(values)});
    $('specialist-dialog').close(); form.reset(); $('dialog-error').textContent = '';
    await loadSpecialists(entry.id); await selectSpecialist(entry.id);
  } catch (error) { $('dialog-error').textContent = error.message; }
};
(async () => {
  try { const id = await loadSpecialists(); await selectSpecialist(id); }
  catch (error) { $('status').textContent = error.message; }
  await loadModels();
})();

let batchRunning = false;
let batchResults = null;
function downloadBatch() {
  if (!batchResults) return;
  const blob = new Blob([JSON.stringify(batchResults, null, 2) + '\n'], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'perguntas-e-respostas.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('batch-open').onclick = () => { if (!busy) $('batch-dialog').showModal(); };
$('batch-close').onclick = () => { if (!batchRunning) $('batch-dialog').close(); };
$('batch-dialog').addEventListener('cancel', event => { if (batchRunning) event.preventDefault(); });
window.addEventListener('beforeunload', event => { if (batchRunning) event.preventDefault(); });
$('batch-download').onclick = downloadBatch;
$('batch-file').onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  try { $('batch-input').value = await file.text(); $('batch-status').textContent = ''; }
  catch (error) { $('batch-status').textContent = `Não foi possível ler o arquivo: ${error.message}`; }
  event.target.value = '';
};
$('batch-form').onsubmit = async event => {
  event.preventDefault();
  if (busy) return;
  let questions;
  try {
    const input = JSON.parse($('batch-input').value);
    questions = input?.perguntas;
    if (!Array.isArray(questions) || !questions.length || questions.some(q => typeof q !== 'string' || !q.trim() || q.trim().length > 4000)) {
      throw new Error('Use um objeto com "perguntas": uma lista não vazia de textos, cada um com até 4000 caracteres.');
    }
    if (!$('model').value) throw new Error('Selecione um modelo do LM Studio antes de começar.');
  } catch (error) { $('batch-status').textContent = `JSON inválido ou dados incompletos: ${error.message}`; return; }
  const specialist = activeSpecialistId, model = $('model').value, thinking = $('thinking').checked;
  batchResults = {especialista: specialist, modelo: model, resultados: []};
  batchRunning = busy = true;
  $('batch-start').disabled = $('batch-close').disabled = $('batch-input').disabled = $('batch-file').disabled = true;
  $('batch-download').hidden = true;
  $('specialist').disabled = $('send').disabled = true;
  try {
    for (let i = 0; i < questions.length; i++) {
      const pergunta = questions[i].trim();
      $('batch-status').textContent = `Respondendo ${i + 1} de ${questions.length}…`;
      try {
        const data = await fetchJSON('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question: pergunta, model, specialist_id: specialist, thinking, conversation_id: null})});
        batchResults.resultados.push({pergunta, resposta: data.answer, conversa_id: data.conversation_id});
        await loadConversations();
      } catch (error) {
        batchResults.resultados.push({pergunta, resposta: null, conversa_id: null, erro: error.message});
      }
    }
    const failures = batchResults.resultados.filter(item => item.erro).length;
    $('batch-status').textContent = `Concluído: ${questions.length - failures} resposta(s), ${failures} falha(s). O arquivo inclui todas as perguntas.`;
    $('batch-download').hidden = false;
    downloadBatch();
  } finally {
    batchRunning = busy = false;
    $('batch-start').disabled = $('batch-close').disabled = $('batch-input').disabled = $('batch-file').disabled = false;
    $('specialist').disabled = $('send').disabled = false;
  }
};
