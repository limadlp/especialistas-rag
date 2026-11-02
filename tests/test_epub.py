from io import BytesIO
from pathlib import Path
import re
import threading
from zipfile import ZipFile, ZIP_STORED

import numpy as np
from PIL import Image
import pytest

from especialistas_rag import epub, nei_gong, specialist_platform


def make_epub(path: Path, *, image_only=False):
    container = '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="EPUB/content.opf"/></rootfiles></container>'
    manifest_extra = '<item id="pic" href="images/page.png" media-type="image/png"/>' if image_only else ''
    opf = f'''<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/">
    <metadata><dc:title>Livro de teste</dc:title><dc:creator>Autora Exemplo</dc:creator></metadata>
    <manifest><item id="one" href="text/one.xhtml" media-type="application/xhtml+xml"/>
    <item id="two" href="text/two.xhtml" media-type="application/xhtml+xml"/>{manifest_extra}</manifest>
    <spine><itemref idref="two"/><itemref idref="one"/></spine></package>'''
    first = '<html><body><h1>Primeiro capítulo</h1><p>' + ('Texto incorporado sobre atenção e prática. ' * 8) + '</p></body></html>'
    second = ('<html><body><h1>Capítulo ilustrado</h1><img src="../images/page.png"/></body></html>' if image_only else
              '<html><body><h1>Segundo capítulo</h1><p>' + ('Texto incorporado sobre memória e estudo. ' * 8) + '</p></body></html>')
    with ZipFile(path, 'w') as archive:
        archive.writestr('mimetype', 'application/epub+zip', compress_type=ZIP_STORED)
        archive.writestr('META-INF/container.xml', container)
        archive.writestr('EPUB/content.opf', opf)
        archive.writestr('EPUB/text/one.xhtml', first)
        archive.writestr('EPUB/text/two.xhtml', second)
        if image_only:
            image = Image.new('RGB', (500, 100), 'white')
            output = BytesIO()
            image.save(output, format='PNG')
            archive.writestr('EPUB/images/page.png', output.getvalue())


class Encoder:
    def tokenizer(self, text, **kwargs):
        return {'offset_mapping': [match.span() for match in re.finditer(r'\S+', text)]}


def test_epub_metadata_spine_order_and_chunk_locations(tmp_path):
    path = tmp_path / 'book.epub'
    make_epub(path)
    book = specialist_platform.book_metadata(path, 'book.epub', 'por')
    assert (book['format'], book['title'], book['author'], book['sections']) == ('epub', 'Livro de teste', 'Autora Exemplo', 2)
    rows = specialist_platform.chunks_for_book(path, book, Encoder())
    assert rows[0]['section_index'] == 1
    assert rows[0]['section_title'] == 'Segundo capítulo'
    assert 'memória' in rows[0]['text']
    assert 'pdf_page' not in rows[0]
    assert 'Segundo capítulo' in nei_gong.source_label(rows[0])
    assert rows[-1]['section_title'] == 'Primeiro capítulo'


def test_image_only_epub_section_uses_ocr(tmp_path):
    path = tmp_path / 'illustrated.epub'
    make_epub(path, image_only=True)
    sections = epub.extract(path, 'por', lambda image, language: 'Texto lido por OCR sobre o tema da obra. ' * 5)
    assert sections[0]['ocr'] is True
    assert 'Texto lido por OCR' in sections[0]['text']
    assert sections[1]['ocr'] is False


def test_rejects_zip_without_epub_package(tmp_path):
    path = tmp_path / 'bad.epub'
    with ZipFile(path, 'w') as archive:
        archive.writestr('other.txt', 'not a book')
    with pytest.raises(ValueError, match='EPUB'):
        specialist_platform.book_metadata(path, 'bad.epub', 'por')


def test_hybrid_search_accepts_epub_section_without_pdf_page():
    row = {'id': 'epub-s0001-c000', 'book_id': 'epub', 'title': 'Livro', 'author': 'Autora',
           'section_index': 1, 'section_title': 'Capítulo um', 'chunk_index': 0,
           'text': 'Atenção e prática em um capítulo do livro.'}

    class Client:
        def query_points(self, *args, **kwargs):
            return type('Result', (), {'points': [type('Hit', (), {'payload': row})()]})()

    class EncoderStub:
        def encode(self, *args, **kwargs):
            return np.array([1.0, 0.0])

    library = nei_gong.Library.__new__(nei_gong.Library)
    library.rows = [row]
    library.by_id = {row['id']: row}
    library.page_limits = {'epub': 0}
    library.books = [{'id': 'epub', 'format': 'epub', 'sections': 1}]
    library.bm25 = None
    library.client = Client()
    library.encoder = EncoderStub()
    library.collection = 'books'
    library.lock = threading.Lock()
    assert library.search(['atenção'], limit=1) == [row]
