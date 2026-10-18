"""Deterministic source transformations. Never execute input code."""

import ast
import io
import tokenize

ATTACKS = ('identity', 'strip_comments', 'reformat', 'strip_docstrings', 'rename_locals')
VERSION = 1


def _rename_locals(tree):
    # Restrict renaming to simple module-level functions; signatures remain intact.
    forbidden = (ast.Global, ast.Nonlocal, ast.Lambda, ast.ClassDef, ast.FunctionDef,
                 ast.AsyncFunctionDef, ast.ListComp, ast.SetComp, ast.DictComp,
                 ast.GeneratorExp, ast.Import, ast.ImportFrom, ast.ExceptHandler,
                 ast.Match, ast.JoinedStr)
    occupied = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    occupied.update(n.arg for n in ast.walk(tree) if isinstance(n, ast.arg))
    changed = False
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        nodes = [n for statement in function.body for n in ast.walk(statement)]
        if any(isinstance(n, forbidden) for n in nodes):
            continue
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id in ('eval', 'exec', 'locals', 'globals', 'vars', 'dir') for n in nodes):
            continue
        parameters = {n.arg for n in ast.walk(function.args) if isinstance(n, ast.arg)}
        candidates = sorted({n.id for n in nodes if isinstance(n, ast.Name)
                             and isinstance(n.ctx, ast.Store)} - parameters)
        mapping = {}
        index = 0
        for name in candidates:
            while f'cg_local_{index}' in occupied:
                index += 1
            replacement = f'cg_local_{index}'
            occupied.add(replacement)
            mapping[name] = replacement
        for node in nodes:
            if isinstance(node, ast.Name) and node.id in mapping:
                node.id = mapping[node.id]
                changed = True
    return changed


def transform(code, attack):
    """Return candidate text and explicit applicability; tests establish correctness."""
    if attack not in ATTACKS:
        raise ValueError(f'Unknown attack: {attack}')
    if code is None:
        return {'code': None, 'status': 'source_unavailable', 'changed': False}
    if attack == 'identity':
        return {'code': code, 'status': 'unchanged', 'changed': False}
    try:
        tree = ast.parse(code)
        if attack == 'strip_comments':
            # Token positions index Unicode characters, unlike AST byte offsets.
            lines = code.splitlines(keepends=True)
            for token in tokenize.generate_tokens(io.StringIO(code).readline):
                if token.type == tokenize.COMMENT:
                    row, start = token.start
                    end = token.end[1]
                    lines[row - 1] = lines[row - 1][:start] + ' ' * (end - start) + lines[row - 1][end:]
            candidate = ''.join(lines)
        else:
            applied = attack == 'reformat'
            if attack == 'strip_docstrings':
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if (node.body and isinstance(node.body[0], ast.Expr)
                                and isinstance(node.body[0].value, ast.Constant)
                                and isinstance(node.body[0].value.value, str)):
                            node.body = node.body[1:] or [ast.Pass()]
                            applied = True
            elif attack == 'rename_locals':
                applied = _rename_locals(tree)
            candidate = ast.unparse(ast.fix_missing_locations(tree)) + '\n' if applied else code
        # Verify syntax without executing code. The parser alone accepts some invalid scopes.
        compile(candidate, '<robustness-candidate>', 'exec')
        if attack in ('strip_comments', 'reformat'):
            if ast.dump(ast.parse(code)) != ast.dump(ast.parse(candidate)):
                raise ValueError('Transformation changed the syntax tree')
        return {'code': candidate, 'status': 'changed' if candidate != code else 'unchanged',
                'changed': candidate != code}
    except (SyntaxError, ValueError, tokenize.TokenError, RecursionError) as exc:
        return {'code': None, 'status': 'transform_error', 'changed': False, 'detail': str(exc)}
