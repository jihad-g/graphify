"""GDScript (Godot) extractor + cross-file resolver (tree-sitter-gdscript).

Two halves, same split as the Terraform extractor:

``extract_gdscript(path)`` / ``extract_godot_scene(path)``
    Per-file. Emits the file, its ``class_name`` class, inner classes,
    functions, signals and enums, plus every same-file call / emit / connect.
    Anything that needs another file to resolve (autoloads, ``class_name``
    types, ``preload`` aliases, inherited methods, typed receivers) is parked
    in ``result["gdscript"]`` for the resolver.

``resolve_gdscript(per_file, all_nodes, all_edges)``
    Cross-file. Reads ``project.godot`` (autoload singletons), indexes every
    script by its ``res://`` path and ``class_name``, then links:

    * ``extends``                       -> ``inherits``
    * ``preload`` / ``load``            -> ``imports``
    * ``Autoload.f()`` / ``Class.f()``  -> ``calls`` (walks the extends chain)
    * ``super.f()`` / inherited ``f()`` -> ``calls``
    * ``typed_var.f()``                 -> ``calls`` (``var x: T`` declarations)
    * ``X.new()``                       -> ``instantiates``
    * ``Bus.sig.emit()``                -> ``emits``      (caller -> signal)
    * ``Bus.sig.connect(handler)``      -> ``listened_by`` (signal -> handler)
    * type annotations / ``is`` / ``as`` -> ``references``
    * ``.tscn`` scripts / sub-scenes    -> ``uses`` / ``instantiates``
    * duck-typed ``obj.f()``            -> ``calls`` INFERRED, only when ``f`` is
      defined exactly once in the project (same single-definition guard the
      other member-call resolvers use).

Nodes carry ``_gd_qual`` (``file``, ``class``, ``cls:Inner``, ``fn:f``,
``fn:Inner.f``, ``signal:hit``, ``enum:State``) so the resolver can find them
again by ``(res path, qual)`` after id remapping and on incremental rebuilds,
where unchanged files arrive only as persisted graph.json nodes.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from graphify.extractors._godot_api import GODOT_METHODS
from graphify.extractors.base import _file_stem, _make_id, _read_source_bytes, _read_text

_LOG = logging.getLogger(__name__)

_SIGNAL_VERBS = frozenset({"emit", "connect", "disconnect", "is_connected"})
# Engine callbacks and very generic names: a duck-typed `x.update()` must not
# bind to whichever single script happens to define `update`.
_DUCK_SKIP = frozenset({
    "new", "free", "queue_free", "call", "call_deferred", "callv", "get", "set",
    "has", "has_method", "is_valid", "duplicate", "clear", "size", "append",
    "erase", "find", "keys", "values", "push_back", "pop_back", "emit",
    "connect", "disconnect", "show", "hide", "start", "stop", "play", "update",
    "reset", "setup", "apply", "open", "close", "add", "remove", "build",
    "load", "save", "init", "refresh", "tick", "toggle", "enable", "disable",
})


# --------------------------------------------------------------------------
# project helpers
# --------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _find_project_root(directory: str) -> str | None:
    """Nearest ancestor directory holding project.godot, or None."""
    d = Path(directory)
    for cand in (d, *d.parents):
        if (cand / "project.godot").is_file():
            return str(cand)
    return None


def _res_rel(path: Path, project_root: str | None) -> str:
    """Project-relative POSIX path (what `res://` names), or the file name."""
    if project_root:
        try:
            return path.resolve().relative_to(Path(project_root).resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def _strip_res(s: str) -> str | None:
    s = s.strip().strip("\"'")
    if s.startswith("res://"):
        return s[len("res://"):]
    return None


def _parse_project_godot(project_root: str) -> dict:
    """Autoload singletons and main scene from project.godot."""
    out: dict[str, Any] = {"autoloads": {}, "main_scene": None}
    try:
        text = (Path(project_root) / "project.godot").read_text(
            encoding="utf-8", errors="replace")
    except OSError:
        return out
    section = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if "=" not in line or line.startswith(";"):
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"')
        if section == "autoload":
            rel = _strip_res(val.lstrip("*"))
            if rel:
                out["autoloads"][key] = rel
        elif section == "application" and key == "run/main_scene":
            out["main_scene"] = _strip_res(val)
    return out


# --------------------------------------------------------------------------
# per-file: .gd
# --------------------------------------------------------------------------

def _first_named(node, type_: str):
    return next((c for c in node.named_children if c.type == type_), None)


def _type_name(node, source: bytes) -> str | None:
    """`type` node -> the class name it names (`Array[Foo]` -> Foo)."""
    if node is None:
        return None
    text = _read_text(node, source)
    m = re.findall(r"[A-Za-z_]\w*", text)
    if not m:
        return None
    # Array[Foo] / Dictionary[String, Foo]: the innermost user type is the useful one
    if m[0] in ("Array", "Dictionary") and len(m) > 1:
        return m[-1]
    return m[0]


def extract_gdscript(path: Path) -> dict:
    """Extract classes, functions, signals, enums and calls from a .gd file."""
    try:
        import tree_sitter_gdscript as tsgd
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_gdscript not installed"}

    try:
        parser = Parser(Language(tsgd.language()))
        source = _read_source_bytes(path)
        root = parser.parse(source).root_node
    except Exception as e:  # pragma: no cover - parser failure
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    project_root = _find_project_root(str(path.resolve().parent))
    res = _res_rel(path, project_root)

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(nid: str, label: str, line: int, qual: str, **extra) -> str:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": str_path, "source_location": f"L{line}",
                          "_gd_qual": qual, **extra})
        return nid

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", context: str | None = None) -> None:
        if src == tgt:
            return
        key = (src, tgt, relation)
        if key in seen_edges:
            return
        seen_edges.add(key)
        e = {"source": src, "target": tgt, "relation": relation,
             "confidence": confidence, "source_file": str_path,
             "source_location": f"L{line}", "weight": 1.0}
        if context:
            e["context"] = context
        edges.append(e)

    file_nid = add_node(_make_id(str(path)), path.name, 1, "file")

    meta: dict[str, Any] = {
        "project_root": project_root, "res": res, "source_file": str_path,
        "class_name": None, "extends": [],   # [(owner_qual, kind, value, line)]
        "loads": [],          # [(res_path, line)]
        "member_calls": [],   # [(caller_qual, receiver, callee, line)]
        "bare_calls": [],     # [(caller_qual, owner_qual, callee, line)]
        "signal_uses": [],    # [(caller_qual, receiver, signal, verb, handler, line)]
        "type_refs": [],      # [(type_name, line)]
        "var_types": {},      # owner_qual -> {var_name: type_name}
        "aliases": {},        # const Name = preload("res://x") -> {Name: res}
        "enum_names": [],
    }

    # owner_qual: "" for the script itself, "Inner" for an inner class
    funcs: dict[str, dict[str, str]] = {}    # owner_qual -> {fname: nid}
    signals: dict[str, str] = {}             # name -> nid (script-level + inner)
    owner_nid: dict[str, str] = {"": file_nid}
    bodies: list[tuple[str, str, Any]] = []  # (caller_qual, owner_qual, body)

    def qual_of(owner: str, name: str) -> str:
        return f"{owner}.{name}" if owner else name

    def collect_loads(node) -> None:
        """preload()/load() string args anywhere under node."""
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type == "call":
                fn = _first_named(n, "identifier")
                if fn is not None and _read_text(fn, source) in ("preload", "load"):
                    args = _first_named(n, "arguments")
                    s = _first_named(args, "string") if args is not None else None
                    if s is not None:
                        rel = _strip_res(_read_text(s, source))
                        if rel:
                            meta["loads"].append((rel, n.start_point[0] + 1))
            stack.extend(n.named_children)

    def record_var(owner: str, node) -> None:
        name = node.child_by_field_name("name") or _first_named(node, "name")
        typ = node.child_by_field_name("type") or _first_named(node, "type")
        if name is None:
            return
        vname = _read_text(name, source)
        tname = _type_name(typ, source) if typ is not None else None
        if tname is None:
            # `var p := Foo.new()` / `var p = Foo.new()` infers Foo
            for c in node.named_children:
                if c.type == "attribute":
                    ids = [x for x in c.named_children]
                    if (len(ids) >= 2 and ids[0].type == "identifier"
                            and ids[-1].type == "attribute_call"):
                        callee = _first_named(ids[-1], "identifier")
                        if callee is not None and _read_text(callee, source) == "new":
                            tname = _read_text(ids[0], source)
        if tname:
            meta["var_types"].setdefault(owner, {})[vname] = tname
            if tname[:1].isupper():
                meta["type_refs"].append((tname, node.start_point[0] + 1))

    def walk_decls(node, owner: str) -> None:
        parent_nid = owner_nid[owner]
        for child in node.named_children:
            t = child.type
            line = child.start_point[0] + 1
            if t == "class_name_statement":
                nm = _first_named(child, "name")
                if nm is not None and not owner:
                    cname = _read_text(nm, source)
                    meta["class_name"] = cname
                    cnid = add_node(_make_id(stem, cname), cname, line, "class",
                                    _callable_class=True)
                    add_edge(file_nid, cnid, "contains", line)
                    owner_nid[""] = cnid
                    parent_nid = cnid
                    # an `extends` seen before class_name stays attached to "" -> fine
            elif t == "extends_statement":
                s = _first_named(child, "string")
                if s is not None:
                    rel = _strip_res(_read_text(s, source))
                    if rel:
                        meta["extends"].append((owner, "path", rel, line))
                else:
                    tn = _type_name(_first_named(child, "type"), source)
                    if tn:
                        meta["extends"].append((owner, "name", tn, line))
            elif t == "signal_statement":
                nm = _first_named(child, "name")
                if nm is not None:
                    sname = _read_text(nm, source)
                    snid = add_node(_make_id(stem, owner, "signal", sname),
                                    f"signal {sname}", line, f"signal:{qual_of(owner, sname)}")
                    signals.setdefault(sname, snid)
                    add_edge(parent_nid, snid, "defines", line)
            elif t == "enum_definition":
                nm = _first_named(child, "name")
                if nm is not None:
                    ename = _read_text(nm, source)
                    enid = add_node(_make_id(stem, owner, ename), ename, line,
                                    f"enum:{qual_of(owner, ename)}")
                    add_edge(parent_nid, enid, "contains", line)
                    meta["enum_names"].append(qual_of(owner, ename))
                    lst = _first_named(child, "enumerator_list")
                    for en in (lst.named_children if lst is not None else []):
                        idn = _first_named(en, "identifier")
                        if idn is None:
                            continue
                        mname = _read_text(idn, source)
                        mnid = add_node(_make_id(enid, mname), mname, en.start_point[0] + 1,
                                        f"enum:{qual_of(owner, ename)}.{mname}")
                        add_edge(enid, mnid, "case_of", en.start_point[0] + 1)
            elif t == "class_definition":
                nm = _first_named(child, "name")
                if nm is None:
                    continue
                iname = qual_of(owner, _read_text(nm, source))
                inid = add_node(_make_id(stem, iname), iname.split(".")[-1], line,
                                f"cls:{iname}", _callable_class=True)
                add_edge(parent_nid, inid, "contains", line)
                owner_nid[iname] = inid
                ext = _first_named(child, "extends_statement")
                if ext is not None:
                    s = _first_named(ext, "string")
                    if s is not None and (rel := _strip_res(_read_text(s, source))):
                        meta["extends"].append((iname, "path", rel, line))
                    else:
                        tn = _type_name(_first_named(ext, "type"), source)
                        if tn:
                            meta["extends"].append((iname, "name", tn, line))
                body = _first_named(child, "class_body")
                if body is not None:
                    walk_decls(body, iname)
            elif t in ("function_definition", "constructor_definition"):
                nm = _first_named(child, "name")
                fname = _read_text(nm, source) if nm is not None else "_init"
                fq = qual_of(owner, fname)
                fnid = add_node(_make_id(stem, owner, fname), f".{fname}()", line, f"fn:{fq}",
                                _callable=True)
                funcs.setdefault(owner, {}).setdefault(fname, fnid)
                add_edge(parent_nid, fnid, "method", line)
                rtype = _type_name(_first_named(child, "type"), source)
                if rtype and rtype[:1].isupper():
                    meta["type_refs"].append((rtype, line))
                params = _first_named(child, "parameters")
                for p in (params.named_children if params is not None else []):
                    pt = _first_named(p, "type")
                    pn = _first_named(p, "identifier") or _first_named(p, "name")
                    tn = _type_name(pt, source) if pt is not None else None
                    if tn and pn is not None:
                        meta["var_types"].setdefault(fq, {})[_read_text(pn, source)] = tn
                        if tn[:1].isupper():
                            meta["type_refs"].append((tn, line))
                body = _first_named(child, "body")
                if body is not None:
                    bodies.append((fq, owner, body))
            elif t in ("variable_statement", "const_statement", "export_variable_statement",
                       "onready_variable_statement"):
                record_var(owner, child)
                collect_loads(child)
                if t == "const_statement":
                    nm = _first_named(child, "name")
                    call = _first_named(child, "call")
                    if nm is not None and call is not None:
                        fn = _first_named(call, "identifier")
                        args = _first_named(call, "arguments")
                        s = _first_named(args, "string") if args is not None else None
                        if (fn is not None and _read_text(fn, source) in ("preload", "load")
                                and s is not None):
                            rel = _strip_res(_read_text(s, source))
                            if rel:
                                meta["aliases"][_read_text(nm, source)] = rel

    walk_decls(root, "")

    # ---- function bodies -------------------------------------------------
    def local_func(owner: str, name: str) -> str | None:
        return funcs.get(owner, {}).get(name)

    def handle_attribute(node, caller: str, owner: str, caller_nid: str) -> None:
        parts = node.named_children
        if not parts:
            return
        line = node.start_point[0] + 1
        head = parts[0]
        receiver = _read_text(head, source) if head.type == "identifier" else None
        # Bus.sig.emit(...) / own_sig.connect(handler)
        if parts[-1].type == "attribute_call":
            verb_node = _first_named(parts[-1], "identifier")
            verb = _read_text(verb_node, source) if verb_node is not None else ""
            mids = [p for p in parts[1:-1]]
            if verb in _SIGNAL_VERBS and receiver is not None:
                handler = None
                args = _first_named(parts[-1], "arguments")
                if args is not None and args.named_children:
                    a0 = args.named_children[0]
                    if a0.type == "identifier":
                        handler = _read_text(a0, source)
                    elif a0.type == "attribute":
                        # self._on_x  /  obj.method
                        ids = [x for x in a0.named_children if x.type == "identifier"]
                        if ids and _read_text(ids[0], source) == "self" and len(ids) > 1:
                            handler = _read_text(ids[-1], source)
                if len(mids) == 1 and mids[0].type == "identifier":
                    # receiver.sig.verb()
                    sig = _read_text(mids[0], source)
                    if receiver == "self" and sig in signals:
                        _own_signal(caller_nid, owner, sig, verb, handler, line)
                    else:
                        meta["signal_uses"].append((caller, receiver, sig, verb, handler, line))
                    return
                if not mids and receiver in signals:
                    # own_sig.verb()
                    _own_signal(caller_nid, owner, receiver, verb, handler, line)
                    return
                if not mids:
                    # sig declared in a base script: resolver walks the chain
                    meta["signal_uses"].append((caller, None, receiver, verb, handler, line))
                    return
            callee = verb
            if len(parts) == 2 and receiver is not None:
                if receiver == "self":
                    tgt = local_func(owner, callee)
                    if tgt:
                        add_edge(caller_nid, tgt, "calls", line)
                    else:
                        meta["bare_calls"].append((caller, owner, callee, line))
                    return
                meta["member_calls"].append((caller, receiver, callee, line))
                return
            # Foo.new().setup(): the first hop still has a known receiver
            if receiver is not None and len(parts) > 2 and parts[1].type == "attribute_call":
                first = _first_named(parts[1], "identifier")
                if first is not None:
                    meta["member_calls"].append((caller, receiver, _read_text(first, source), line))
            # longer chains: a.b.c(), get_tree().x() -> receiver unknown
            if callee:
                meta["member_calls"].append((caller, None, callee, line))
            return
        # plain property access: Autoload.gold / Class.CONST / Alias.Enum
        if receiver and receiver[:1].isupper():
            meta["type_refs"].append((receiver, line))

    def _own_signal(caller_nid, owner, sig, verb, handler, line) -> None:
        snid = signals.get(sig)
        if snid is None:
            return
        if verb == "emit":
            add_edge(caller_nid, snid, "emits", line)
        elif verb in ("connect", "disconnect"):
            tgt = (local_func(owner, handler) if handler else None) or caller_nid
            add_edge(snid, tgt, "listened_by", line)

    def walk_body(node, caller: str, owner: str, caller_nid: str) -> None:
        stack = [node]
        while stack:
            n = stack.pop()
            t = n.type
            if t == "attribute":
                handle_attribute(n, caller, owner, caller_nid)
            elif t == "call":
                fn = _first_named(n, "identifier")
                if fn is not None:
                    name = _read_text(fn, source)
                    line = n.start_point[0] + 1
                    if name in ("preload", "load"):
                        collect_loads(n)
                    elif name == "emit_signal":
                        args = _first_named(n, "arguments")
                        s = _first_named(args, "string") if args is not None else None
                        if s is not None:
                            _own_signal(caller_nid, owner, _read_text(s, source).strip("\"'"),
                                        "emit", None, line)
                    else:
                        tgt = local_func(owner, name)
                        if tgt:
                            add_edge(caller_nid, tgt, "calls", line)
                        else:
                            meta["bare_calls"].append((caller, owner, name, line))
            elif t == "variable_statement":
                record_var(caller, n)
            elif t == "binary_operator":
                # x is Foo / x as Foo
                kids = n.named_children
                txt = _read_text(n, source)
                if len(kids) == 2 and re.search(r"\b(is|as)\b", txt) and kids[1].type == "identifier":
                    tn = _read_text(kids[1], source)
                    if tn[:1].isupper():
                        meta["type_refs"].append((tn, n.start_point[0] + 1))
            elif t == "type":
                tn = _type_name(n, source)
                if tn and tn[:1].isupper():
                    meta["type_refs"].append((tn, n.start_point[0] + 1))
            elif t == "identifier" and n.parent is not None and n.parent.type not in ("attribute", "call"):
                # bare identifier passed as a value: `Foo` used as a type/value
                tn = _read_text(n, source)
                if tn[:1].isupper() and not tn.isupper():
                    meta["type_refs"].append((tn, n.start_point[0] + 1))
            stack.extend(reversed(n.named_children))

    for caller, owner, body in bodies:
        caller_nid = funcs.get(owner, {}).get(caller.split(".")[-1]) or owner_nid[owner]
        walk_body(body, caller, owner, caller_nid)

    meta["funcs"] = {o: sorted(fs) for o, fs in funcs.items()}
    meta["signals"] = sorted(signals)
    return {"nodes": nodes, "edges": edges, "raw_calls": [], "gdscript": meta}


# --------------------------------------------------------------------------
# per-file: .tscn scenes
# --------------------------------------------------------------------------

_EXT_RES_RE = re.compile(r"^\[ext_resource\b([^\]]*)\]")
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def extract_godot_scene(path: Path) -> dict:
    """A .tscn scene: one node, plus its Script / PackedScene ext_resources."""
    str_path = str(path)
    project_root = _find_project_root(str(path.resolve().parent))
    res = _res_rel(path, project_root)
    try:
        text = _read_source_bytes(path).decode("utf-8", errors="replace")
    except Exception as e:  # pragma: no cover
        return {"nodes": [], "edges": [], "error": str(e)}
    nid = _make_id(str(path))
    nodes = [{"id": nid, "label": path.name, "file_type": "code",
              "source_file": str_path, "source_location": "L1", "_gd_qual": "file"}]
    edges: list[dict] = []
    refs: list[tuple[str, str, int]] = []
    root_done = False
    for i, line in enumerate(text.splitlines(), 1):
        if not root_done and line.startswith("[node "):
            # the scene's root node (the only one without a parent= attribute)
            attrs = dict(_ATTR_RE.findall(line))
            if "parent" not in attrs and attrs.get("name"):
                rid = _make_id(_file_stem(path), "root", attrs["name"])
                label = attrs["name"] + (f" ({attrs['type']})" if attrs.get("type") else "")
                nodes.append({"id": rid, "label": label, "file_type": "code",
                              "source_file": str_path, "source_location": f"L{i}",
                              "_gd_qual": "scene_root"})
                edges.append({"source": nid, "target": rid, "relation": "contains",
                              "confidence": "EXTRACTED", "source_file": str_path,
                              "source_location": f"L{i}", "weight": 1.0})
                root_done = True
            continue
        m = _EXT_RES_RE.match(line)
        if not m:
            continue
        attrs = dict(_ATTR_RE.findall(m.group(1)))
        rel = _strip_res(attrs.get("path", ""))
        if rel:
            refs.append((attrs.get("type", ""), rel, i))
    meta = {"project_root": project_root, "res": res, "source_file": str_path,
            "scene_refs": refs}
    return {"nodes": nodes, "edges": edges, "raw_calls": [], "gdscript": meta}


# --------------------------------------------------------------------------
# cross-file resolver
# --------------------------------------------------------------------------

def _node_res(source_file: str, project_root: str, known: set[str]) -> str | None:
    """Map a node's source_file (absolute on fresh nodes, scan-root relative on
    persisted ones) back to its res:// relative path."""
    sf = str(source_file).replace("\\", "/")
    root = str(Path(project_root).resolve()).replace("\\", "/").rstrip("/") + "/"
    if sf.startswith(root):
        return sf[len(root):]
    if sf in known:
        return sf
    for k in known:
        if sf.endswith("/" + k):
            return k
    return None


def resolve_gdscript(per_file: list[dict], all_nodes: list[dict], all_edges: list[dict]) -> None:
    metas = [r["gdscript"] for r in per_file if isinstance(r, dict) and r.get("gdscript")]
    by_project: dict[str, list[dict]] = {}
    for m in metas:
        if m.get("project_root"):
            by_project.setdefault(m["project_root"], []).append(m)
    for project_root, pmetas in by_project.items():
        try:
            _resolve_project(project_root, pmetas, all_nodes, all_edges)
        except Exception as exc:  # keep the build alive, like other resolvers
            _LOG.warning("GDScript resolution failed for %s: %s", project_root, exc)


def _resolve_project(project_root: str, metas: list[dict],
                     all_nodes: list[dict], all_edges: list[dict]) -> None:
    proj = _parse_project_godot(project_root)
    autoloads: dict[str, str] = proj["autoloads"]

    # Every script/scene file the project has, so persisted (relative) nodes
    # can be mapped back to res paths too.
    known: set[str] = {m["res"] for m in metas}
    root_path = Path(project_root)
    for ext in ("*.gd", "*.tscn"):
        for p in root_path.rglob(ext):
            if ".godot" in p.parts or ".git" in p.parts:
                continue
            known.add(p.relative_to(root_path).as_posix())

    # (res, qual) -> node id
    idx: dict[tuple[str, str], str] = {}
    node_by_id: dict[str, dict] = {}
    class_names: dict[str, str] = {}
    for n in all_nodes:
        q = n.get("_gd_qual")
        if not q:
            continue
        r = _node_res(n.get("source_file", ""), project_root, known)
        if r is None:
            continue
        idx[(r, q)] = n["id"]
        node_by_id[n["id"]] = n
        if q == "class":
            class_names[n.get("label", "")] = r
    for name, r in autoloads.items():
        fid = idx.get((r, "file"))
        if fid and fid in node_by_id:
            node_by_id[fid]["autoload"] = name
    for m in metas:
        if m.get("class_name"):
            class_names[m["class_name"]] = m["res"]

    # quals per script, for method / signal lookup through the extends chain
    funcs_of: dict[str, dict[str, set[str]]] = {}
    signals_of: dict[str, set[str]] = {}
    for (r, q) in idx:
        if q.startswith("signal:"):
            signals_of.setdefault(r, set()).add(q[len("signal:"):])
        elif q.startswith("fn:"):
            owner, _, fn = q[3:].rpartition(".")
            funcs_of.setdefault(r, {}).setdefault(owner, set()).add(fn)

    def script_nid(r: str, owner: str = "") -> str | None:
        if owner:
            return idx.get((r, f"cls:{owner}"))
        return idx.get((r, "class")) or idx.get((r, "file"))

    def name_to_res(name: str | None, m: dict) -> str | None:
        if not name:
            return None
        if name in m.get("aliases", {}):
            return m["aliases"][name]
        if name in class_names:
            return class_names[name]
        if name in autoloads:
            return autoloads[name]
        return None

    # extends chain: (res, owner) -> (res, owner) of the base script
    base_of: dict[tuple[str, str], tuple[str, str]] = {}
    for m in metas:
        for owner, kind, value, _line in m.get("extends", []):
            if kind == "path":
                base_of[(m["res"], owner)] = (value, "")
            else:
                if value in m.get("enum_names", []):
                    continue
                if (m["res"], f"cls:{value}") in idx and value != owner:
                    base_of[(m["res"], owner)] = (m["res"], value)   # inner class base
                elif (r := name_to_res(value, m)):
                    base_of[(m["res"], owner)] = (r, "")
    # persisted inherits edges for unchanged files (incremental builds)
    rev = {nid: key for key, nid in idx.items()}
    for e in all_edges:
        if e.get("relation") != "inherits":
            continue
        s, t = rev.get(e.get("source")), rev.get(e.get("target"))
        if s and t:
            if not all(k[1] in ("file", "class") or k[1].startswith("cls:") for k in (s, t)):
                continue
            s_owner = s[1][4:] if s[1].startswith("cls:") else ""
            t_owner = t[1][4:] if t[1].startswith("cls:") else ""
            base_of.setdefault((s[0], s_owner), (t[0], t_owner))

    def find_in_chain(r: str, owner: str, name: str, table: str) -> tuple[str, str] | None:
        """(res, qual) of `name` defined in (r, owner) or any base script."""
        seen = set()
        cur: tuple[str, str] | None = (r, owner)
        while cur and cur not in seen:
            seen.add(cur)
            cr, co = cur
            if table == "func":
                if name in funcs_of.get(cr, {}).get(co, set()):
                    return cr, "fn:" + (f"{co}.{name}" if co else name)
            else:
                q = f"{co}.{name}" if co else name
                if q in signals_of.get(cr, set()):
                    return cr, f"signal:{q}"
            cur = base_of.get(cur)
        return None

    # unique-name index for duck-typed calls
    defs_by_name: dict[str, list[tuple[str, str]]] = {}
    for r, owners in funcs_of.items():
        for owner, fns in owners.items():
            for fn in fns:
                defs_by_name.setdefault(fn, []).append((r, "fn:" + (f"{owner}.{fn}" if owner else fn)))

    existing = {(e.get("source"), e.get("target"), e.get("relation")) for e in all_edges}

    def add(src: str | None, tgt: str | None, relation: str, m: dict, line: int,
            confidence: str = "EXTRACTED", context: str | None = None) -> None:
        if not src or not tgt or src == tgt:
            return
        key = (src, tgt, relation)
        if key in existing:
            return
        existing.add(key)
        e = {"source": src, "target": tgt, "relation": relation,
             "confidence": confidence, "source_file": m["source_file"],
             "source_location": f"L{line}", "weight": 1.0}
        if context:
            e["context"] = context
        all_edges.append(e)

    def caller_id(m: dict, caller: str) -> str | None:
        return idx.get((m["res"], f"fn:{caller}")) or script_nid(m["res"])

    def owner_of(caller: str) -> str:
        return caller.rpartition(".")[0]

    def receiver_target(m: dict, caller: str, receiver: str | None) -> tuple[str, str] | None:
        """(res, owner) the receiver resolves to, or None."""
        if receiver is None:
            return None
        owner = owner_of(caller)
        if receiver == "self":
            return m["res"], owner
        if receiver == "super":
            return base_of.get((m["res"], owner))
        if (m["res"], f"cls:{receiver}") in idx:
            return m["res"], receiver          # inner class used by name
        if (r := name_to_res(receiver, m)):
            return r, ""
        # typed variable: function scope, then owner class, then script
        for scope in (caller, owner, ""):
            tn = m["var_types"].get(scope, {}).get(receiver)
            if tn:
                if (m["res"], f"cls:{tn}") in idx:
                    return m["res"], tn
                if (r := name_to_res(tn, m)):
                    return r, ""
                return None
        return None

    for m in metas:
        res = m["res"]
        file_nid = idx.get((res, "file"))

        # scenes
        for typ, rel, line in m.get("scene_refs", []):
            tgt = script_nid(rel) if rel.endswith(".gd") else idx.get((rel, "file"))
            if tgt:
                add(file_nid, tgt, "uses" if typ == "Script" else "instantiates", m, line)
        if "scene_refs" in m:
            continue

        for (owner, base_res_owner) in [(k[1], v) for k, v in base_of.items() if k[0] == res]:
            src = script_nid(res, owner)
            tgt = script_nid(*base_res_owner)
            line = next((ln for o, _k, _v, ln in m["extends"] if o == owner), 1)
            add(src, tgt, "inherits", m, line)

        for rel, line in m["loads"]:
            tgt = idx.get((rel, "file"))
            add(file_nid, tgt, "imports", m, line, context="preload")

        for tn, line in m["type_refs"]:
            r = name_to_res(tn, m)
            if r and r != res:
                add(file_nid, script_nid(r), "references", m, line)

        for caller, owner, callee, line in m["bare_calls"]:
            base = base_of.get((res, owner))
            if not base:
                continue
            hit = find_in_chain(base[0], base[1], callee, "func")
            if hit:
                add(caller_id(m, caller), idx.get(hit), "calls", m, line, context="inherited")

        for caller, receiver, callee, line in m["member_calls"]:
            src = caller_id(m, caller)
            tgt_script = receiver_target(m, caller, receiver)
            if tgt_script:
                if callee == "new":
                    add(src, script_nid(*tgt_script), "instantiates", m, line)
                    continue
                hit = find_in_chain(tgt_script[0], tgt_script[1], callee, "func")
                if hit:
                    add(src, idx.get(hit), "calls", m, line)
                elif tgt_script[0] != res:
                    add(file_nid, script_nid(tgt_script[0]), "references", m, line)
                continue
            if callee in _DUCK_SKIP or callee in GODOT_METHODS or callee.startswith("_"):
                continue
            cands = defs_by_name.get(callee, [])
            if len(cands) == 1 and cands[0][0] != res:
                add(src, idx.get(cands[0]), "calls", m, line,
                    confidence="INFERRED", context="duck_typed")

        for caller, receiver, sig, verb, handler, line in m["signal_uses"]:
            src = caller_id(m, caller)
            owner = owner_of(caller)
            tgt_script = (receiver_target(m, caller, receiver) if receiver
                          else (res, owner))
            if not tgt_script:
                continue
            hit = find_in_chain(tgt_script[0], tgt_script[1], sig, "signal")
            if not hit:
                continue
            snid = idx.get(hit)
            if verb == "emit":
                add(src, snid, "emits", m, line)
            elif verb in ("connect", "disconnect"):
                hnid = None
                if handler:
                    h = find_in_chain(res, owner, handler, "func")
                    hnid = idx.get(h) if h else None
                add(snid, hnid or src, "listened_by", m, line)
