"""Read EPUB spine content without extracting archive members to disk."""
from __future__ import annotations

from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
import posixpath
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

from PIL import Image


OPF = "http://www.idpf.org/2007/opf"
DC = "http://purl.org/dc/elements/1.1/"
MAX_ARCHIVE_BYTES = 600 * 1024 * 1024
MAX_MEMBERS = 10000
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024


def _read(archive: ZipFile, name: str, limit: int = MAX_DOCUMENT_BYTES) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise ValueError(f"EPUB incompleto: falta {name}") from exc
    if info.file_size > limit:
        raise ValueError(f"Arquivo interno do EPUB muito grande: {name}")
    return archive.read(info)


def _path(base: str, href: str) -> str:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        raise ValueError("Recurso externo não pode ser indexado")
    name = unquote(parsed.path).lstrip("/")
    result = posixpath.normpath(posixpath.join(posixpath.dirname(base), name))
    if result in ("", ".", "..") or result.startswith("../"):
        raise ValueError("Caminho inválido dentro do EPUB")
    return result


def package(path: Path) -> tuple[dict, list[dict]]:
    try:
        with ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_MEMBERS or sum(item.file_size for item in members) > MAX_ARCHIVE_BYTES:
                raise ValueError("EPUB expandido excede o limite de segurança")
            if _read(archive, "mimetype", 100).strip() != b"application/epub+zip":
                raise ValueError("Arquivo EPUB inválido: tipo de arquivo ausente")
            container = ET.fromstring(_read(archive, "META-INF/container.xml", 1_000_000))
            rootfile = container.find(".//{*}rootfile")
            if rootfile is None or not rootfile.get("full-path"):
                raise ValueError("EPUB sem documento de pacote")
            opf_path = rootfile.get("full-path")
            opf = ET.fromstring(_read(archive, opf_path))
            title_node = opf.find(f".//{{{DC}}}title")
            creator_node = opf.find(f".//{{{DC}}}creator")
            title = "".join(title_node.itertext()).strip() if title_node is not None else ""
            author = "".join(creator_node.itertext()).strip() if creator_node is not None else ""
            manifest = {}
            for item in opf.findall(f".//{{{OPF}}}manifest/{{{OPF}}}item"):
                if item.get("id") and item.get("href"):
                    manifest[item.get("id")] = item
            entries = []
            for itemref in opf.findall(f".//{{{OPF}}}spine/{{{OPF}}}itemref"):
                if itemref.get("linear", "yes") == "no":
                    continue
                item = manifest.get(itemref.get("idref"))
                if item is None or "nav" in item.get("properties", "").split():
                    continue
                media_type = item.get("media-type", "")
                if media_type not in ("application/xhtml+xml", "text/html"):
                    continue
                entry_path = _path(opf_path, item.get("href"))
                archive.getinfo(entry_path)
                entries.append({"path": entry_path})
            if not entries:
                raise ValueError("EPUB sem capítulos XHTML na ordem de leitura")
            return {"title": title, "author": author}, entries
    except (BadZipFile, ET.ParseError, KeyError) as exc:
        raise ValueError(f"EPUB inválido: {exc}") from exc


class _Content(HTMLParser):
    BLOCKS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "blockquote", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.images: list[str] = []
        self.heading = ""
        self._heading_parts: list[str] | None = None
        self._skip: str | None = None

    def handle_starttag(self, tag, attrs):
        tag = tag.split(":")[-1].lower()
        if self._skip:
            return
        if tag in ("head", "script", "style", "nav"):
            self._skip = tag
            return
        attributes = dict(attrs)
        if tag in ("img", "image"):
            source = attributes.get("src") or attributes.get("href") or attributes.get("xlink:href")
            if source:
                self.images.append(source)
        if tag in ("h1", "h2", "h3") and not self.heading:
            self._heading_parts = []
        if tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.split(":")[-1].lower()
        if self._skip:
            if tag == self._skip:
                self._skip = None
            return
        if tag in ("h1", "h2", "h3") and self._heading_parts is not None:
            candidate = re.sub(r"\s+", " ", "".join(self._heading_parts)).strip()
            if candidate and not self.heading:
                self.heading = candidate[:180]
            self._heading_parts = None
        if tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)
            if self._heading_parts is not None:
                self._heading_parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts).replace("\xa0", " ")
        value = re.sub(r"[ \t]+", " ", value)
        return re.sub(r"\n\s*\n+", "\n\n", value).strip()


def extract(path: Path, language: str, ocr_image, progress=None) -> list[dict]:
    _, entries = package(path)
    result = []
    with ZipFile(path) as archive:
        for number, entry in enumerate(entries, 1):
            source = entry["path"]
            parser = _Content()
            parser.feed(_read(archive, source).decode("utf-8-sig", errors="replace"))
            text = parser.text()
            used_ocr = False
            if len(re.sub(r"\W", "", text)) < 80:
                fragments = []
                for href in parser.images:
                    try:
                        image_path = _path(source, href)
                        with Image.open(BytesIO(_read(archive, image_path, 60 * 1024 * 1024))) as image:
                            fragment = ocr_image(image.convert("RGB"), language).strip()
                            if fragment:
                                fragments.append(fragment)
                    except (ValueError, KeyError, OSError):
                        continue
                if fragments:
                    text = "\n\n".join(fragments)
                    used_ocr = True
            result.append({"section_index": number, "section_title": parser.heading or Path(source).stem,
                           "text": text, "ocr": used_ocr})
            if progress:
                progress(number, len(entries))
    return result
