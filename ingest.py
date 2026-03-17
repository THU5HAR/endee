"""
Stage 1 — Data Ingestion
Clone GitHub repositories and extract functions/methods via tree-sitter AST parsing.
Output: data/<repo_name>.jsonl (one JSON object per parsed function)
"""

import os
import json
import uuid
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
from tree_sitter import Language, Parser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPOS_DIR = Path(os.getenv("REPOS_DIR", "data/repos"))
DATA_DIR = Path("data")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ingest")

# Repositories to index (GitHub org/repo format)
DEFAULT_REPOS = [
    "scikit-learn/scikit-learn",
    "tiangolo/fastapi",
    "psf/requests",
    "pallets/flask",
    "numpy/numpy",
    "pandas-dev/pandas",
    "django/django",
    "sqlalchemy/sqlalchemy",
    "pydantic/pydantic",
    "encode/httpx",
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeSnippet:
    id: str
    repo: str
    filepath: str
    function_name: str
    language: str
    start_line: int
    end_line: int
    code: str
    docstring: str


# ---------------------------------------------------------------------------
# tree-sitter setup (tree-sitter 0.21.x API)
# ---------------------------------------------------------------------------

PY_LANGUAGE = Language(tspython.language(), "python")
JS_LANGUAGE = Language(tsjavascript.language(), "javascript")


def _make_parser(language: Language) -> Parser:
    p = Parser()
    p.set_language(language)
    return p


PY_PARSER = _make_parser(PY_LANGUAGE)
JS_PARSER = _make_parser(JS_LANGUAGE)


# ---------------------------------------------------------------------------
# Git clone
# ---------------------------------------------------------------------------

def clone_repo(repo_slug: str, dest_dir: Path) -> Optional[Path]:
    """Clone a GitHub repo to dest_dir/<repo_name>. Returns the cloned path."""
    repo_name = repo_slug.split("/")[-1]
    target = dest_dir / repo_name

    if target.exists():
        log.info("Repo '%s' already cloned at %s — skipping clone.", repo_name, target)
        return target

    url = f"https://github.com/{repo_slug}.git"
    log.info("Cloning %s → %s ...", url, target)
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "--single-branch", url, str(target)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        log.info("Cloned '%s' successfully.", repo_name)
        return target
    except subprocess.CalledProcessError as exc:
        log.error("Failed to clone '%s': %s", repo_slug, exc.stderr.decode())
        return None
    except subprocess.TimeoutExpired:
        log.error("Clone timeout for '%s'.", repo_slug)
        return None


# ---------------------------------------------------------------------------
# Python AST parsing
# ---------------------------------------------------------------------------

def _extract_docstring_python(node, source_bytes: bytes) -> str:
    """Extract the docstring from a Python function body if present."""
    body = None
    for child in node.children:
        if child.type == "block":
            body = child
            break
    if body is None:
        return ""
    # First expression statement → string
    for stmt in body.children:
        if stmt.type == "expression_statement":
            for child in stmt.children:
                if child.type == "string":
                    raw = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                    return raw.strip("\"'").strip()
    return ""


def parse_python_functions(filepath: Path, repo_name: str) -> list[CodeSnippet]:
    """Parse all function/method definitions in a Python file."""
    snippets: list[CodeSnippet] = []
    try:
        source_bytes = filepath.read_bytes()
        tree = PY_PARSER.parse(source_bytes)
    except Exception as exc:
        log.debug("Parse error in %s: %s", filepath, exc)
        return snippets

    def traverse(node):
        if node.type in ("function_definition", "decorated_definition"):
            target = node
            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type == "function_definition":
                        target = child
                        break
            name_node = None
            for child in target.children:
                if child.type == "identifier":
                    name_node = child
                    break
            if name_node is None:
                for child in node.children:
                    traverse(child)
                return

            func_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
            code = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            docstring = _extract_docstring_python(target, source_bytes)

            # Relative path from repo root
            try:
                rel_path = str(filepath.relative_to(REPOS_DIR / repo_name))
            except ValueError:
                rel_path = str(filepath)

            snippets.append(CodeSnippet(
                id=str(uuid.uuid4()),
                repo=repo_name,
                filepath=rel_path,
                function_name=func_name,
                language="python",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                code=code[:4000],  # cap at 4k chars to avoid huge embeddings
                docstring=docstring[:500],
            ))
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return snippets


# ---------------------------------------------------------------------------
# JavaScript AST parsing
# ---------------------------------------------------------------------------

def parse_js_functions(filepath: Path, repo_name: str) -> list[CodeSnippet]:
    """Parse function declarations and arrow functions in a JS/TS file."""
    snippets: list[CodeSnippet] = []
    try:
        source_bytes = filepath.read_bytes()
        tree = JS_PARSER.parse(source_bytes)
    except Exception as exc:
        log.debug("Parse error in %s: %s", filepath, exc)
        return snippets

    FUNC_TYPES = {
        "function_declaration",
        "function_expression",
        "method_definition",
        "arrow_function",
    }

    def traverse(node):
        if node.type in FUNC_TYPES:
            func_name = "<anonymous>"
            for child in node.children:
                if child.type == "identifier":
                    func_name = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                    break
                if child.type == "property_identifier":
                    func_name = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                    break

            code = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            try:
                rel_path = str(filepath.relative_to(REPOS_DIR / repo_name))
            except ValueError:
                rel_path = str(filepath)

            snippets.append(CodeSnippet(
                id=str(uuid.uuid4()),
                repo=repo_name,
                filepath=rel_path,
                function_name=func_name,
                language="javascript",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                code=code[:4000],
                docstring="",
            ))
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return snippets


# ---------------------------------------------------------------------------
# Repo-level ingestion
# ---------------------------------------------------------------------------

def parse_repo(repo_path: Path, repo_name: str) -> list[CodeSnippet]:
    """Walk all Python and JS files in a repo and extract functions."""
    all_snippets: list[CodeSnippet] = []
    error_count = 0

    py_files = list(repo_path.rglob("*.py"))
    js_files = list(repo_path.rglob("*.js")) + list(repo_path.rglob("*.ts"))

    log.info("[%s] Parsing %d Python files, %d JS/TS files ...",
             repo_name, len(py_files), len(js_files))

    for f in py_files:
        # Skip test files and vendored code to keep index clean
        if any(part in f.parts for part in (".git", "node_modules", "__pycache__", "migrations")):
            continue
        try:
            snippets = parse_python_functions(f, repo_name)
            all_snippets.extend(snippets)
        except Exception as exc:
            log.warning("Error parsing %s: %s", f, exc)
            error_count += 1

    for f in js_files:
        if any(part in f.parts for part in (".git", "node_modules", "dist", "build")):
            continue
        try:
            snippets = parse_js_functions(f, repo_name)
            all_snippets.extend(snippets)
        except Exception as exc:
            log.warning("Error parsing %s: %s", f, exc)
            error_count += 1

    log.info("[%s] Extracted %d functions (%d parse errors).", repo_name, len(all_snippets), error_count)
    return all_snippets


def save_snippets(snippets: list[CodeSnippet], output_path: Path) -> None:
    """Save snippets to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in snippets:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    log.info("Saved %d snippets to %s", len(snippets), output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ingest_repos(repo_list: Optional[list[str]] = None) -> int:
    """
    Clone and parse all repos.
    Returns total number of snippets ingested.
    """
    if repo_list is None:
        repo_list = DEFAULT_REPOS

    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for repo_slug in repo_list:
        repo_name = repo_slug.split("/")[-1]
        log.info("=== Processing repo: %s ===", repo_slug)

        repo_path = clone_repo(repo_slug, REPOS_DIR)
        if repo_path is None:
            log.error("Skipping '%s' due to clone failure.", repo_slug)
            continue

        snippets = parse_repo(repo_path, repo_name)
        if not snippets:
            log.warning("No functions extracted from '%s'.", repo_name)
            continue

        output_path = DATA_DIR / f"{repo_name}.jsonl"
        save_snippets(snippets, output_path)
        total += len(snippets)

    log.info("=== Ingestion complete. Total functions: %d ===", total)
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest GitHub repos for code search indexing.")
    parser.add_argument(
        "--repos",
        nargs="*",
        default=None,
        help="List of GitHub repo slugs (e.g. scikit-learn/scikit-learn). "
             "Defaults to 10 curated repos.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing clones and re-clone from scratch.",
    )
    args = parser.parse_args()

    if args.clean and REPOS_DIR.exists():
        log.info("--clean flag set: removing %s", REPOS_DIR)
        shutil.rmtree(REPOS_DIR)

    ingest_repos(args.repos)
