/* Small, offline Markdown renderer for model answers. All model text is escaped. */
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[char]);
}

function sourceLocation(source) {
  if (source.section_index != null) {
    return `seção ${source.section_index}: ${source.section_title || 'Sem título'} · trecho ${(source.chunk_index || 0) + 1}`;
  }
  return `página PDF ${source.pdf_page}`;
}

function renderInline(value, sources) {
  let html = escapeHtml(value);
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
  html = html.replace(/(?<!_)_([^_]+)_(?!_)/g, '<em>$1</em>');
  html = html.replace(/\[((?:S\d+)(?:\s*,\s*S\d+)*)\]/g, (_, group) => {
    return group.split(/\s*,\s*/).map(code => {
      const source = sources.get(code);
      if (!source) return `[${code}]`;
      const label = escapeHtml(`${source.author} — ${source.title}, ${sourceLocation(source)}`);
      const excerpt = escapeHtml(source.text);
      return `<button type="button" class="citation" aria-label="Ver fonte ${code}: ${label}" aria-expanded="false">[${code}]<span class="citation-preview" role="tooltip"><strong>${label}</strong><span>${excerpt}</span></span></button>`;
    }).join(', ');
  });
  return html;
}

function renderAnswer(markdown, sourceList = []) {
  const sources = new Map(sourceList.map(source => [source.citation, source]));
  const lines = String(markdown).replace(/\r\n?/g, '\n').split('\n');
  const parts = [];
  let paragraph = [];
  let list = null;
  function closeParagraph() {
    if (paragraph.length) {
      parts.push(`<p>${paragraph.map(line => renderInline(line, sources)).join('<br>')}</p>`);
      paragraph = [];
    }
  }
  function closeList() {
    if (list) { parts.push(`</${list}>`); list = null; }
  }
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      closeParagraph(); closeList();
      const code = [];
      while (++i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i]);
      parts.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`);
      continue;
    }
    if (!line.trim()) { closeParagraph(); closeList(); continue; }
    const heading = line.match(/^\s*(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      closeParagraph(); closeList();
      const level = Math.min(heading[1].length + 1, 6);
      parts.push(`<h${level}>${renderInline(heading[2], sources)}</h${level}>`);
      continue;
    }
    const bullet = line.match(/^\s*[-*+]\s+(.+)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      closeParagraph();
      const kind = bullet ? 'ul' : 'ol';
      if (list !== kind) { closeList(); parts.push(`<${kind}>`); list = kind; }
      parts.push(`<li>${renderInline((bullet || numbered)[1], sources)}</li>`);
      continue;
    }
    const quote = line.match(/^\s*>\s?(.*)$/);
    if (quote) {
      closeParagraph(); closeList();
      parts.push(`<blockquote>${renderInline(quote[1], sources)}</blockquote>`);
      continue;
    }
    closeList();
    paragraph.push(line.trim());
  }
  closeParagraph(); closeList();
  const container = document.createElement('div');
  container.className = 'answer-markdown';
  container.innerHTML = parts.join('');
  container.querySelectorAll('.citation').forEach(button => {
    const preview = button.querySelector('.citation-preview');
    const placePreview = () => {
      preview.style.visibility = 'hidden';
      preview.style.display = 'block';
      const bounds = button.getBoundingClientRect();
      const width = preview.getBoundingClientRect().width;
      const height = preview.getBoundingClientRect().height;
      const left = Math.max(12, Math.min(bounds.left, window.innerWidth - width - 12));
      const top = bounds.top - height - 9 >= 12
        ? bounds.top - height - 9
        : Math.min(bounds.bottom + 9, window.innerHeight - height - 12);
      preview.style.left = `${left}px`;
      preview.style.top = `${Math.max(12, top)}px`;
      preview.style.display = '';
      preview.style.visibility = '';
    };
    button.addEventListener('mouseenter', placePreview);
    button.addEventListener('focus', placePreview);
    button.addEventListener('click', event => {
      event.stopPropagation();
      placePreview();
      const open = button.classList.toggle('is-open');
      button.setAttribute('aria-expanded', String(open));
    });
  });
  return container;
}

document.addEventListener('click', () => {
  document.querySelectorAll('.citation.is-open').forEach(button => {
    button.classList.remove('is-open');
    button.setAttribute('aria-expanded', 'false');
  });
});
