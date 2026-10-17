"""Bounded, conservative dataflow from installed forward source, without execution.

Only straight-line tensor dependencies and branches resolved from primitive module
configuration are expanded. Dynamic control flow stays inside an opaque module.
No model code is evaluated, traced, patched or run to construct these diagrams.
"""

import ast
import inspect
import logging
import textwrap
from functools import lru_cache

from torch.nn import Linear
from transformers.pytorch_utils import Conv1D

from circuit_lantern.services.fp8 import is_fp8_linear

_UNKNOWN = object()


class UnresolvedFlow(Exception):
    """Source cannot prove the selected path at this module boundary."""


def projection_module(module):
    """A parameter named weight alone does not prove a projection uses it."""
    if type(module) in (Linear, Conv1D) or is_fp8_linear(module):
        return True
    if type(module).__module__ != "bitsandbytes.nn.modules":
        return False
    from bitsandbytes.nn import Linear4bit, Linear8bitLt, LinearFP4, LinearNF4

    return type(module) in (Linear4bit, Linear8bitLt, LinearFP4, LinearNF4)


@lru_cache(maxsize=256)
def forward_tree(cls):
    try:
        source = inspect.getsource(inspect.unwrap(cls.forward))
        if len(source) > 64_000:
            return None
        tree = ast.parse(textwrap.dedent(source))
        if sum(1 for _ in ast.walk(tree)) > 8000:
            return None
        return next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    except (
        OSError,
        TypeError,
        SyntaxError,
        StopIteration,
        AttributeError,
        RecursionError,
        ValueError,
    ):
        return None


def primitive(node, module, constants):
    """Resolve inert values only. Never call getters or tensor truth operations."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id, _UNKNOWN)
    if isinstance(node, ast.Attribute):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name) or node.id != "self":
            return _UNKNOWN
        if parts == ["training"]:
            return False
        value = module
        for part in reversed(parts):
            value = vars(value).get(part, _UNKNOWN) if hasattr(value, "__dict__") else _UNKNOWN
        return value if type(value) in (str, bool, int, float, type(None)) else _UNKNOWN
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = primitive(node.operand, module, constants)
        return not value if type(value) is bool else _UNKNOWN
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = (primitive(n, module, constants) for n in [node.left, node.comparators[0]])
        if left is _UNKNOWN or right is _UNKNOWN:
            return _UNKNOWN
        op = node.ops[0]
        if isinstance(op, ast.Is):
            return left is right
        if isinstance(op, ast.IsNot):
            return left is not right
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
    return _UNKNOWN


class SourceFlow:
    def __init__(self, weights, expand=None):
        self.weights = weights
        self.expand = expand
        self.nodes = []
        self.edges = []
        self.opaque = []
        self.mixer_type = None

    def node(self, label, inputs=(), *, weight=None, detail="", section="block", **extra):
        if len(self.nodes) >= 180:
            raise UnresolvedFlow
        identifier = "op-" + str(len(self.nodes))
        self.nodes.append(
            dict(id=identifier, label=label, detail=detail, weight=weight, section=section, **extra)
        )
        for source in dict.fromkeys(inputs):
            self.edges.append(
                {"source": source, "target": identifier, "label": "", "residual": False}
            )
        return (identifier,)

    def call(self, module, path, inputs, depth=0, *, can_expand=True):
        if self.expand and can_expand:
            result = self.expand(self, module, path, inputs)
            if result is not None:
                return result
        weight = path + ".weight"
        if weight in self.weights and projection_module(module):
            return self.node(
                path.rsplit(".", 1)[-1],
                inputs,
                weight=weight,
                detail=" × ".join(map(str, self.weights[weight]["shape"])),
            )
        start = len(self.nodes), len(self.edges), len(self.opaque), self.mixer_type
        if can_expand and depth < 3 and list(module.children()):
            try:
                return self.compile(module, path, inputs, depth)
            except UnresolvedFlow:
                del self.nodes[start[0] :]
                del self.edges[start[1] :]
                del self.opaque[start[2] :]
                self.mixer_type = start[3]
        descendants = [name for name in self.weights if name.startswith(path + ".")]
        if descendants:
            self.opaque.append(path)
        return self.node(
            type(module).__name__,
            inputs,
            detail=path,
            matrices=descendants,
            opaque=bool(descendants),
        )

    def compile(self, module, path, inputs, depth=0):
        tree = forward_tree(type(module))
        if tree is None:
            raise UnresolvedFlow
        arguments = [
            arg.arg
            for arg in tree.args.posonlyargs + tree.args.args + tree.args.kwonlyargs
            if arg.arg != "self"
        ]
        if not arguments:
            raise UnresolvedFlow
        env = {arguments[0]: inputs}
        constants = {}
        # This diagram describes uncached inference, never a cached decode path.
        for name in ("past_key_value", "past_key_values", "cache_params"):
            constants[name] = None
        children = dict(module.named_modules())

        def expression(node):
            if isinstance(node, ast.Name):
                if node.id in env:
                    return env[node.id]
                if node.id in constants or node.id in arguments:
                    return ()
                raise UnresolvedFlow
            if isinstance(node, ast.Constant):
                return ()
            if (
                isinstance(node, ast.Attribute)
                and primitive(node, module, constants) is not _UNKNOWN
            ):
                return ()
            if isinstance(node, (ast.Tuple, ast.List)):
                # Structured assignment handles tuples element by element.
                # Flattening here would invent dependencies across tuple ports.
                raise UnresolvedFlow
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
                refs = expression(node.left) + expression(node.right)
                if not refs:
                    return ()
                return self.node("Add" if isinstance(node.op, ast.Add) else "Multiply", refs)
            if isinstance(node, ast.Call):
                target = node.func
                names = []
                while isinstance(target, ast.Attribute):
                    names.append(target.attr)
                    target = target.value
                if not isinstance(target, ast.Name) or target.id != "self":
                    raise UnresolvedFlow
                relative = ".".join(reversed(names))
                child = children.get(relative)
                if child is None or not relative:
                    raise UnresolvedFlow
                # Control arguments are not activation edges. Additional tensor
                # arguments remain opaque rather than silently disappearing.
                args = [expression(arg) for arg in node.args]
                keywords = {
                    kw.arg: expression(kw.value) for kw in node.keywords if kw.arg is not None
                }
                child_tree = forward_tree(type(child))
                first = (
                    next(
                        (
                            arg.arg
                            for arg in child_tree.args.posonlyargs
                            + child_tree.args.args
                            + child_tree.args.kwonlyargs
                            if arg.arg != "self"
                        ),
                        None,
                    )
                    if child_tree
                    else None
                )
                primary = args[0] if args else keywords.get(first, ())
                args += list(keywords.values())
                refs = tuple(ref for group in args for ref in group)
                if not refs:
                    raise UnresolvedFlow
                before = len(self.nodes)
                result = self.call(
                    child,
                    path + "." + relative,
                    refs,
                    depth + 1,
                    can_expand=bool(primary) and sum(bool(group) for group in args) == 1,
                )
                for item in self.nodes[before:]:
                    item.setdefault("module_path", path + "." + relative)
                    if relative == "mlp":
                        item["section"] = "mlp"
                return result
            raise UnresolvedFlow

        def assign(target, refs):
            if isinstance(target, ast.Name):
                env[target.id] = refs
                constants.pop(target.id, None)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for i, child in enumerate(target.elts):
                    if not isinstance(child, ast.Name):
                        raise UnresolvedFlow
                    if child.id != "_":
                        constants.pop(child.id, None)
                        env[child.id] = (
                            self.node("Module output", refs, detail=str(i + 1)) if refs else ()
                        )
            else:
                raise UnresolvedFlow

        def statements(body):
            for statement in body:
                if isinstance(statement, ast.Assign):
                    if (
                        len(statement.targets) == 1
                        and isinstance(statement.targets[0], (ast.Tuple, ast.List))
                        and isinstance(statement.value, (ast.Tuple, ast.List))
                    ):
                        targets = statement.targets[0].elts
                        if len(targets) != len(statement.value.elts):
                            raise UnresolvedFlow
                        values = [expression(item) for item in statement.value.elts]
                        for target, refs in zip(targets, values, strict=True):
                            assign(target, refs)
                        continue
                    refs = expression(statement.value)
                    for target in statement.targets:
                        assign(target, refs)
                elif isinstance(statement, ast.AugAssign) and isinstance(
                    statement.op, (ast.Add, ast.Mult)
                ):
                    refs = expression(statement.target) + expression(statement.value)
                    assign(
                        statement.target,
                        self.node("Add" if isinstance(statement.op, ast.Add) else "Multiply", refs)
                        if refs
                        else (),
                    )
                elif isinstance(statement, ast.AnnAssign) and statement.value:
                    assign(statement.target, expression(statement.value))
                elif isinstance(statement, ast.If):
                    value = primitive(statement.test, module, constants)
                    if type(value) is not bool:
                        raise UnresolvedFlow
                    result = statements(statement.body if value else statement.orelse)
                    if result is not None:
                        return result
                elif isinstance(statement, ast.Return):
                    # Causal blocks conventionally return hidden states first.
                    value = statement.value
                    if isinstance(value, ast.Tuple):
                        if not value.elts:
                            raise UnresolvedFlow
                        value = value.elts[0]
                    return expression(value)
                elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                    continue
                else:
                    raise UnresolvedFlow
            return None

        result = statements(tree.body)
        if not result:
            raise UnresolvedFlow
        return result

    def finish(self, layer, block):
        # Only the returned primary stream is drawn. Unused tuple outputs or
        # discarded computations must not be labelled as feeding the next block.
        live = {self.nodes[-1]["id"]}
        for node in reversed(self.nodes):
            if node["id"] in live:
                live.update(edge["source"] for edge in self.edges if edge["target"] == node["id"])
        self.nodes = [node for node in self.nodes if node["id"] in live]
        self.edges = [
            edge for edge in self.edges if edge["source"] in live and edge["target"] in live
        ]
        self.opaque = list(
            dict.fromkeys(node["detail"] for node in self.nodes if node.get("opaque"))
        )
        # Layered DAG positions keep actual branches side by side, never an
        # invented sequence inferred from module registration order.
        levels = {}
        lanes = {}
        for node in self.nodes:
            sources = [edge["source"] for edge in self.edges if edge["target"] == node["id"]]
            level = max((levels[source] + 1 for source in sources), default=0)
            levels[node["id"]] = level
            lanes.setdefault(level, []).append(node)
        for level, nodes in lanes.items():
            for i, node in enumerate(nodes):
                node.update(x=240 + (i - (len(nodes) - 1) / 2) * 230, y=level * 115)
        # A residual shortcut must be proven by reachability to the other input
        # of an add, rather than inferred from module or variable names.
        ancestors = {}
        for node in self.nodes:
            incoming = [edge for edge in self.edges if edge["target"] == node["id"]]
            ancestors[node["id"]] = (
                set().union(*(ancestors[edge["source"]] | {edge["source"]} for edge in incoming))
                if incoming
                else set()
            )
            if node["label"] == "Add":
                for edge in incoming:
                    if any(
                        edge["source"] in ancestors[other["source"]]
                        for other in incoming
                        if other is not edge
                    ):
                        edge.update(residual=True, label="residual")
        shown = {node["weight"] for node in self.nodes if node["weight"]}
        return {
            "layer": layer,
            "supported": True,
            "nodes": self.nodes,
            "edges": self.edges,
            "mixer_type": self.mixer_type,
            "provenance": "source",
            "block_type": type(block).__name__,
            "opaque_modules": self.opaque,
            "unmapped_matrices": [name for name in self.weights if name not in shown],
            "assumptions": [
                "Evaluation mode",
                "Uncached inference",
                "Optional tensor inputs and extra keyword inputs are summarized",
                "Primary block output is the first tuple element",
                "Module hooks and decorators may alter execution",
            ],
            "mapped_matrices": len(shown),
            "total_matrices": len(self.weights),
            "scope": "Source-derived module dependencies for uncached inference. Opaque modules retain their weight inventory without claiming internal connections. This is not measured causal attribution.",
        }


def source_block_dataflow(block, prefix, weights, layer, expand=None):
    graph = SourceFlow(weights, expand)
    try:
        inputs = graph.node("Residual input")
        outputs = graph.compile(block, prefix.rstrip("."), inputs)
        graph.node("Residual output", outputs)
        return graph.finish(layer, block)
    except UnresolvedFlow:
        return None
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not derive block dataflow for %s", type(block).__name__, exc_info=True
        )
        return None
