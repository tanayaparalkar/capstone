"""Patch extraction and generation from AI suggestions or security knowledge base."""
import difflib
import re
from pathlib import Path
from typing import Optional, Tuple

from sentinelai.contracts import AIEnrichedFinding, ScannerFinding
from .models import Patch


def _strip_markdown_code_fences(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[0].strip().startswith("```"):
        lines.pop(0)
    if lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines)


def _align_indentation(orig: str, repl: str) -> str:
    orig_lines = orig.splitlines()
    repl_lines = repl.splitlines()
    if not orig_lines or not repl_lines:
        return repl
    orig_indent = len(orig_lines[0]) - len(orig_lines[0].lstrip())
    repl_indent = len(repl_lines[0]) - len(repl_lines[0].lstrip())
    if repl_indent == 0 and orig_indent > 0:
        indent_str = " " * orig_indent
        return "\n".join(indent_str + line if line.strip() else line for line in repl_lines)
    return repl


def _parse_diff_lines(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse diff lines with '-' and '+' prefixes into original and replacement text."""
    lines = text.splitlines()
    has_diff_prefix = any(line.startswith(("+", "-")) for line in lines if line.strip())
    if not has_diff_prefix:
        return None, None

    orig_lines = []
    repl_lines = []
    for line in lines:
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("-"):
            orig_lines.append(line[1:])
        elif line.startswith("+"):
            repl_lines.append(line[1:])
        else:
            orig_lines.append(line)
            repl_lines.append(line)

    return "\n".join(orig_lines), "\n".join(repl_lines)


def _generate_template_patch(
    finding: ScannerFinding,
    file_content: str,
    line_start: Optional[int],
    line_end: Optional[int],
) -> Optional[Tuple[int, int, str, str, str]]:
    """Fallback deterministic patches for standard vulnerabilities when AI patch is absent."""
    lines = file_content.splitlines()
    if not lines:
        return None

    l_start = max(1, line_start or 1)
    l_end = min(len(lines), line_end or l_start)

    # 1. SQL Injection: string concat
    if finding.category == "sql-injection" or "sql" in (finding.cwe or "").lower() or "sql" in finding.rule_id.lower():
        # Scan within window around finding lines
        search_start = max(0, l_start - 2)
        search_end = min(len(lines), l_end + 3)
        for idx in range(search_start, search_end):
            line = lines[idx]
            if ("SELECT" in line or "WHERE" in line) and "+" in line:
                var_match = re.search(r'WHERE\s+(\w+)\s*=\s*[\'"]+\s*\+\s*(\w+)\s*\+\s*[\'"]+', line)
                if var_match:
                    field = var_match.group(1)
                    var = var_match.group(2)
                    quote_char = '"' if '="' in line or '("' in line or '"' in line[:line.find("WHERE")] else "'"
                    new_line = re.sub(
                        r'WHERE\s+\w+\s*=\s*[\'"]+\s*\+\s*\w+\s*\+\s*[\'"]+',
                        f'WHERE {field} = ?{quote_char}',
                        line,
                    )
                else:
                    var_match = re.search(r'\+\s*([a-zA-Z_]\w*)\s*(?:\+|;|\)|$)', line)
                    var = var_match.group(1) if var_match else "username"
                    new_line = re.sub(r'WHERE\s+(\w+)\s*=\s*[\'"]?\s*\+\s*\w+\s*(\+\s*[\'"])?', r'WHERE \1 = ?"', line)

                t_start = idx + 1
                t_end = t_start
                orig_block = [line]
                new_block = [new_line]

                # Check if execute(query) is on subsequent lines
                for j in range(idx + 1, min(len(lines), idx + 4)):
                    if "execute(" in lines[j]:
                        t_end = j + 1
                        orig_block = lines[idx:t_end]
                        exec_line = lines[j]
                        new_exec = re.sub(r'execute\((\w+)\)', rf'execute(\1, ({var},))', exec_line)
                        new_block = [new_line] + lines[idx + 1 : j] + [new_exec]
                        break

                return t_start, t_end, "\n".join(orig_block), "\n".join(new_block), "Parameterized SQL query to prevent injection"

    target_slice = lines[l_start - 1 : l_end]
    target_text = "\n".join(target_slice)

    # 2. Insecure Hash: MD5 -> SHA-256 (CWE-327)
    if "md5" in target_text.lower() or finding.category == "weak-cryptography" or "327" in (finding.cwe or ""):
        if "hashlib.md5(" in target_text:
            new_text = target_text.replace("hashlib.md5(", "hashlib.sha256(")
            return l_start, l_end, target_text, new_text, "Upgraded broken MD5 hashing to secure SHA-256"

    # 3. Insecure Deserialization: pickle -> json or safe loading (CWE-502)
    if "pickle.loads(" in target_text:
        new_text = target_text.replace("pickle.loads(blob)", "json.loads(blob.decode('utf-8'))")
        if new_text != target_text:
            return l_start, l_end, target_text, new_text, "Replaced unsafe pickle deserialization with safe json.loads"

    # 4. Unsafe YAML load -> safe_load (CWE-502)
    if "yaml.load(" in target_text:
        new_text = target_text.replace("yaml.load(", "yaml.safe_load(")
        return l_start, l_end, target_text, new_text, "Used yaml.safe_load to prevent arbitrary code execution"

    # 5. Insecure subprocess shell=True (CWE-78)
    if "shell=True" in target_text:
        new_text = re.sub(r"subprocess\.call\(([\w]+),\s*shell=True\)", r"subprocess.call(shlex.split(\1), shell=False)", target_text)
        if new_text != target_text:
            return l_start, l_end, target_text, new_text, "Disabled shell=True and tokenized command arguments safely"

    # 6. os.system with concatenation (CWE-78)
    if "os.system(" in target_text and "+" in target_text:
        match = re.search(r'os\.system\((["\'].*?)\s*\+\s*(\w+)\)', target_text)
        if match:
            cmd_prefix = match.group(1).strip("'\"").strip()
            var = match.group(2)
            parts = [f'"{p}"' for p in cmd_prefix.split()] + [var]
            indent = re.match(r"^(\s*)", target_text).group(1) if target_text else ""
            new_text = f"{indent}subprocess.run([{', '.join(parts)}], check=True)"
            return l_start, l_end, target_text, new_text, "Replaced os.system shell invocation with safe subprocess.run"

    # 7. Hardcoded Secrets (CWE-798)
    sec_match = re.search(r'(\b[A-Za-z0-9_]*(?:KEY|PASSWORD|SECRET|TOKEN)[A-Za-z0-9_]*)\s*=\s*([\'"][^\'"]+[\'"])', target_text)
    if sec_match or finding.category in ("hardcoded-secret", "secret-exposure") or "798" in (finding.cwe or ""):
        if sec_match:
            var_name = sec_match.group(1)
            new_text = re.sub(
                r'(\b' + re.escape(var_name) + r')\s*=\s*([\'"][^\'"]+[\'"])',
                rf'\1 = os.environ.get("{var_name}", "")',
                target_text,
            )
            if new_text != target_text:
                return l_start, l_end, target_text, new_text, f"Replaced hardcoded {var_name} with environment variable lookup"

    # 8. Security Misconfiguration: Debug Mode (CWE-16)
    if "DEBUG = True" in target_text or finding.category == "security-misconfiguration" or "16" in (finding.cwe or ""):
        if "DEBUG = True" in target_text:
            new_text = target_text.replace("DEBUG = True", 'DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1")')
            return l_start, l_end, target_text, new_text, "Disabled hardcoded DEBUG mode in favor of production-safe environment guard"

    # 9. Arbitrary Code Execution: eval -> ast.literal_eval (CWE-94)
    if "eval(" in target_text and "ast.literal_eval(" not in target_text:
        new_text = re.sub(r'\beval\(', 'ast.literal_eval(', target_text)
        if new_text != target_text:
            return l_start, l_end, target_text, new_text, "Replaced dangerous eval() with safe ast.literal_eval()"

    # 10. Cross-Site Scripting (XSS): unescaped output (CWE-79)
    if finding.category == "xss" or "79" in (finding.cwe or ""):
        if "html.escape(" not in target_text:
            # Check for f-string or string concatenation embedding variable in HTML tag
            xss_match = re.search(r'f[\'"].*?<(\w+)>\{(\w+)\}</\1>[\'"]', target_text)
            if xss_match:
                tag = xss_match.group(1)
                var = xss_match.group(2)
                new_text = re.sub(
                    rf'\{{{var}\}}',
                    rf'{{html.escape({var})}}',
                    target_text,
                )
                return l_start, l_end, target_text, new_text, f"Sanitized user input '{var}' using html.escape() to prevent XSS"

    # 11. Broken Access Control / IDOR (CWE-639 / CWE-284)
    if finding.category in ("broken-access-control", "idor") or "639" in (finding.cwe or "") or "284" in (finding.cwe or ""):
        if "owner_id" not in target_text and "current_user" not in target_text:
            match = re.search(r'(\.filter_by\([^)]*id\s*=\s*\w+)(\))', target_text)
            if match:
                new_text = target_text[:match.start(2)] + ", owner_id=current_user.id" + target_text[match.start(2):]
                return l_start, l_end, target_text, new_text, "Scoped database query to current_user.id to prevent IDOR / broken access control"

    # 12. Server-Side Request Forgery (SSRF) (CWE-918)
    if finding.category == "ssrf" or "918" in (finding.cwe or ""):
        if "requests.get(" in target_text and "validate_url" not in target_text:
            new_text = re.sub(r'requests\.get\((\w+)\)', r'validate_safe_url(\1) and requests.get(\1)', target_text)
            if new_text != target_text:
                return l_start, l_end, target_text, new_text, "Added URL scheme and host validation to prevent SSRF"

    # 13. Cross-Site Request Forgery (CSRF) (CWE-352)
    if finding.category == "csrf" or "352" in (finding.cwe or ""):
        if "@csrf_protect" not in target_text and ("methods=['POST']" in target_text or 'methods=["POST"]' in target_text):
            lines_in_target = target_text.splitlines()
            new_lines = []
            for l in lines_in_target:
                if "@app.route" in l:
                    new_lines.append(l)
                    new_lines.append("@csrf_protect")
                else:
                    new_lines.append(l)
            new_text = "\n".join(new_lines)
            if new_text != target_text:
                return l_start, l_end, target_text, new_text, "Applied @csrf_protect decorator to secure state-changing POST endpoint against CSRF"

    return None


def extract_patch(
    repo_root: Path,
    finding: ScannerFinding,
    ai: Optional[AIEnrichedFinding] = None,
) -> Optional[Patch]:
    """Extract or formulate a concrete Patch for a given finding."""
    if not finding.file:
        return None

    target_file = repo_root / finding.file
    if not target_file.exists() or not target_file.is_file():
        return None

    try:
        content = target_file.read_text(encoding="utf-8")
    except Exception:
        return None

    file_lines = content.splitlines()
    line_start = finding.line_start or 1
    line_end = finding.line_end or line_start

    # Determine original snippet
    clamped_start = max(1, min(line_start, len(file_lines)))
    clamped_end = max(clamped_start, min(line_end, len(file_lines)))
    original_snippet = "\n".join(file_lines[clamped_start - 1 : clamped_end])

    replacement_snippet: Optional[str] = None
    explanation = ai.remediation if ai else finding.message

    # Attempt 1: From AI patch_suggestion if available
    if ai and ai.patch_suggestion:
        clean_patch = _strip_markdown_code_fences(ai.patch_suggestion)
        diff_orig, diff_repl = _parse_diff_lines(clean_patch)
        if diff_repl:
            replacement_snippet = _align_indentation(original_snippet, diff_repl)
            if diff_orig and diff_orig.strip():
                # Locate diff_orig in file if lines drifted
                if diff_orig in content:
                    # Find exact line numbers
                    sub_idx = content.find(diff_orig)
                    line_no = content[:sub_idx].count("\n") + 1
                    clamped_start = line_no
                    clamped_end = line_no + diff_orig.count("\n")
                    original_snippet = diff_orig
        else:
            replacement_snippet = _align_indentation(original_snippet, clean_patch)

    # Attempt 2: Knowledge Base / Heuristic template patch
    if not replacement_snippet or replacement_snippet == original_snippet:
        tpl = _generate_template_patch(finding, content, clamped_start, clamped_end)
        if tpl:
            clamped_start, clamped_end, original_snippet, replacement_snippet, tpl_expl = tpl
            if not ai or not ai.remediation:
                explanation = tpl_expl

    if not replacement_snippet or replacement_snippet.strip() == original_snippet.strip():
        return None

    # Compute a clean unified diff for the patch
    orig_diff_lines = [line + "\n" for line in original_snippet.splitlines()]
    repl_diff_lines = [line + "\n" for line in replacement_snippet.splitlines()]
    diff_gen = difflib.unified_diff(
        orig_diff_lines,
        repl_diff_lines,
        fromfile=f"a/{finding.file}",
        tofile=f"b/{finding.file}",
    )
    diff_text = "".join(diff_gen)

    return Patch(
        finding_id=finding.finding_id,
        file_path=target_file,
        line_start=clamped_start,
        line_end=clamped_end,
        original_snippet=original_snippet,
        replacement_snippet=replacement_snippet,
        diff=diff_text,
        explanation=explanation,
        cwe=finding.cwe,
        category=finding.category,
        rule_id=finding.rule_id,
        scanner=finding.scanner,
        severity=finding.severity,
    )

