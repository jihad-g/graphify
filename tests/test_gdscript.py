"""GDScript (Godot) extractor and cross-file resolver."""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path

import pytest

from graphify.extract import extract, extract_gdscript, extract_godot_scene

pytestmark = pytest.mark.skipif(
    _ilu.find_spec("tree_sitter_gdscript") is None,
    reason="tree-sitter-gdscript not installed",
)

PROJECT_GODOT = """\
config_version=5

[application]

run/main_scene="res://scenes/main.tscn"

[autoload]

Events="*res://src/events.gd"
GameState="*res://src/game_state.gd"
"""

EVENTS = """\
extends Node
signal enemy_killed(id)
signal toast(text)
"""

GAME_STATE = """\
extends Node
func add_xp(amount: int) -> void:
	Events.toast.emit("xp")
"""

BASE = """\
class_name Actor
extends CharacterBody3D
signal died
func take_damage(n: int) -> void:
	if n > 0:
		died.emit()
func heal(n: int) -> void:
	pass
"""

PLAYER = """\
class_name Player
extends Actor
const Bolt = preload("res://src/bolt.gd")
var stats: Stats
enum Mode { WALK, RUN }
func _ready() -> void:
	Events.enemy_killed.connect(_on_kill)
	died.connect(_on_died)
	GameState.add_xp(5)
	var b := Bolt.new()
	b.fly()
	stats.recompute()
	take_damage(1)
	super.heal(2)
	var a = [1]
	a.is_empty()
func _on_kill(id) -> void:
	helper()
func _on_died() -> void:
	pass
func helper() -> void:
	pass
class Inner extends Node:
	func inner_fn():
		pass
"""

BOLT = """\
extends Node3D
func fly() -> void:
	pass
func is_empty() -> bool:
	return true
"""

STATS = """\
class_name Stats
extends RefCounted
func recompute() -> void:
	pass
"""

SCENE = """\
[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://src/player.gd" id="1_a"]
[ext_resource type="PackedScene" path="res://scenes/hud.tscn" id="2_b"]

[node name="Main" type="Node3D"]
script = ExtResource("1_a")

[node name="HUD" parent="." instance=ExtResource("2_b")]
"""

HUD = """\
[gd_scene format=3]

[node name="HUD" type="CanvasLayer"]
"""


def _project(tmp_path: Path) -> Path:
    files = {
        "project.godot": PROJECT_GODOT,
        "src/events.gd": EVENTS,
        "src/game_state.gd": GAME_STATE,
        "src/actor.gd": BASE,
        "src/player.gd": PLAYER,
        "src/bolt.gd": BOLT,
        "src/stats.gd": STATS,
        "scenes/main.tscn": SCENE,
        "scenes/hud.tscn": HUD,
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def _graph(tmp_path: Path):
    root = _project(tmp_path)
    paths = sorted(p for p in root.rglob("*") if p.suffix in (".gd", ".tscn"))
    r = extract(paths, cache_root=root, root=root, parallel=False)
    nodes = {n["id"]: n for n in r["nodes"]}

    def name(nid):
        n = nodes[nid]
        return f"{Path(n['source_file']).name}:{n['label']}"

    edges = {(name(e["source"]), e["relation"], name(e["target"]))
             for e in r["edges"] if e["source"] in nodes and e["target"] in nodes}
    return nodes, edges, r


def test_per_file_symbols(tmp_path):
    root = _project(tmp_path)
    r = extract_gdscript(root / "src/player.gd")
    assert "error" not in r
    labels = {n["label"] for n in r["nodes"]}
    assert {"player.gd", "Player", "._ready()", "._on_kill()", "Mode", "WALK",
            "Inner", ".inner_fn()"} <= labels
    meta = r["gdscript"]
    assert meta["class_name"] == "Player"
    assert meta["res"] == "src/player.gd"
    assert meta["aliases"] == {"Bolt": "src/bolt.gd"}


def test_scene_extraction(tmp_path):
    root = _project(tmp_path)
    r = extract_godot_scene(root / "scenes/main.tscn")
    assert {n["label"] for n in r["nodes"]} == {"main.tscn", "Main (Node3D)"}
    assert ("Script", "src/player.gd", 3) in r["gdscript"]["scene_refs"]


def test_cross_file_links(tmp_path):
    _, edges, _ = _graph(tmp_path)
    expected = {
        ("player.gd:Player", "inherits", "actor.gd:Actor"),
        ("player.gd:player.gd", "imports", "bolt.gd:bolt.gd"),
        # autoload method call + signal bus emit / connect
        ("player.gd:._ready()", "calls", "game_state.gd:.add_xp()"),
        ("game_state.gd:.add_xp()", "emits", "events.gd:signal toast"),
        ("events.gd:signal enemy_killed", "listened_by", "player.gd:._on_kill()"),
        # inherited signal + inherited method + super call
        ("actor.gd:signal died", "listened_by", "player.gd:._on_died()"),
        ("player.gd:._ready()", "calls", "actor.gd:.take_damage()"),
        ("player.gd:._ready()", "calls", "actor.gd:.heal()"),
        # preload alias, inferred var type, declared var type
        ("player.gd:._ready()", "instantiates", "bolt.gd:bolt.gd"),
        ("player.gd:._ready()", "calls", "bolt.gd:.fly()"),
        ("player.gd:._ready()", "calls", "stats.gd:.recompute()"),
        ("player.gd:player.gd", "references", "stats.gd:Stats"),
        # same-file
        ("player.gd:._on_kill()", "calls", "player.gd:.helper()"),
        ("actor.gd:.take_damage()", "emits", "actor.gd:signal died"),
        # scenes
        ("main.tscn:main.tscn", "uses", "player.gd:Player"),
        ("main.tscn:main.tscn", "instantiates", "hud.tscn:hud.tscn"),
    }
    missing = expected - edges
    assert not missing, missing


def test_engine_method_names_never_bind_by_name(tmp_path):
    """`a.is_empty()` on an Array must not bind to bolt.gd's is_empty()."""
    _, edges, _ = _graph(tmp_path)
    assert ("player.gd:._ready()", "calls", "bolt.gd:.is_empty()") not in edges


def test_autoload_marked_on_file_node(tmp_path):
    nodes, _, _ = _graph(tmp_path)
    autoloads = {Path(n["source_file"]).name: n.get("autoload")
                 for n in nodes.values() if n.get("autoload")}
    assert autoloads == {"events.gd": "Events", "game_state.gd": "GameState"}


def test_godot_sidecars_not_unclassified(tmp_path):
    from graphify.detect import detect
    root = _project(tmp_path)
    (root / "src/player.gd.uid").write_text("uid://abc")
    (root / "icon.png.import").write_text("[remap]")
    det = detect(root)
    names = {Path(p).name for p in det.get("unclassified", [])}
    assert "player.gd.uid" not in names
    assert "icon.png.import" not in names
