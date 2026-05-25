"""
NOVA Filesystem Tools
---------------------
Ported from Shifu — file read, write, directory listing.
All paths are relative to nova_data/ (the rover's workspace).
"""

import os
from pathlib import Path
from .base import Tool, ToolResult

NOVA_DATA = Path(__file__).resolve().parent.parent / "nova_data"
NOVA_DATA.mkdir(exist_ok=True)


def _resolve(filepath: str) -> Path:
    p = Path(filepath)
    if p.is_absolute():
        return p
    # Strip leading nova_data/ to avoid double-nesting
    try:
        p = p.relative_to("nova_data")
    except ValueError:
        pass
    return NOVA_DATA / p


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a file and return its contents. "
        "Supports text, JSON, CSV, PDF, DOCX, XLSX. "
        "Relative paths resolve to nova_data/."
    )
    parameters = {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to file."},
        },
        "required": ["filepath"],
    }

    async def execute(self, filepath: str) -> ToolResult:
        path = _resolve(filepath)
        if not path.exists():
            return ToolResult(False, f"File not found: {path}")

        ext = path.suffix.lower()

        try:
            if ext == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                text   = "\n\n".join(pg.extract_text() or "" for pg in reader.pages).strip()
                return ToolResult(True, "Read PDF.", text or "(no extractable text)")

            if ext == ".docx":
                from docx import Document
                doc  = Document(str(path))
                text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
                return ToolResult(True, "Read DOCX.", text or "(empty)")

            if ext in (".xlsx", ".xls", ".xlsm"):
                import openpyxl
                wb   = openpyxl.load_workbook(str(path), data_only=True)
                rows = []
                for ws in wb.worksheets:
                    for row in ws.iter_rows(values_only=True):
                        rows.append("\t".join(str(c or "") for c in row))
                return ToolResult(True, "Read spreadsheet.", "\n".join(rows))

            content = path.read_text(encoding="utf-8", errors="replace")
            return ToolResult(True, f"Read {len(content)} chars.", content)

        except ImportError as e:
            return ToolResult(False, f"Missing dependency: {e}")
        except Exception as e:
            return ToolResult(False, f"Read error: {e}")


class WriteFile(Tool):
    name = "write_file"
    description = (
        "Write text to a file. "
        "Relative paths resolve to nova_data/. Parent directories are created."
    )
    parameters = {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Destination path."},
            "content":  {"type": "string", "description": "Text content to write."},
        },
        "required": ["filepath", "content"],
    }

    async def execute(self, filepath: str, content: str) -> ToolResult:
        path = _resolve(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(content, encoding="utf-8")
            return ToolResult(True, f"Written {len(content)} chars to {path}.")
        except Exception as e:
            return ToolResult(False, f"Write error: {e}")


class ListFiles(Tool):
    name = "list_files"
    description = "List files in a directory. Defaults to nova_data/."
    parameters = {
        "type": "object",
        "properties": {
            "dirpath": {"type": "string", "description": "Directory to list."},
        },
    }

    async def execute(self, dirpath: str = ".") -> ToolResult:
        path = _resolve(dirpath)
        if not path.exists():
            return ToolResult(False, f"Directory not found: {path}")
        lines = []
        for item in sorted(path.rglob("*")):
            indent = "  " * (len(item.relative_to(path).parts) - 1)
            marker = "📁" if item.is_dir() else "📄"
            lines.append(f"{indent}{marker} {item.name}")
        summary = "\n".join(lines) if lines else "(empty)"
        return ToolResult(True, f"Listed {path}.", summary)
