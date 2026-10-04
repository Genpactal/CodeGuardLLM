"""Conservative extraction: never repair, reformat, or execute model output."""

import ast
import re


def extract_code(raw, entry_point):
    blocks = re.findall(r"^```(?:python|py)?[ \t]*\r?\n(.*?)^```[ \t]*$", raw, re.MULTILINE | re.DOTALL)
    if "```" in raw and len(blocks) != 1:
        return {"status": "extraction_error", "code": None, "detail": "Expected exactly one complete Python fence"}
    code = blocks[0] if blocks else raw
    # Strip surrounding blank lines only; indentation and internal whitespace matter.
    code = code.strip("\r\n")
    if not code.strip():
        return {"status": "empty", "code": None, "detail": "Empty completion"}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        return {"status": "syntax_error", "code": code, "detail": str(exc)}
    if not any(isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body):
        return {"status": "missing_entrypoint", "code": code, "detail": f"Expected def {entry_point}(...)"}
    return {"status": "ok", "code": code, "detail": None}
