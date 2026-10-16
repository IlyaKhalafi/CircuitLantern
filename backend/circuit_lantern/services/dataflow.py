"""Architecture-specific block dataflow from inspected modules and logical weights."""

SUPPORTED_BLOCKS = {
    "GPT2Block",
    "GPTNeoXLayer",
    "LlamaDecoderLayer",
    "MistralDecoderLayer",
    "Qwen2DecoderLayer",
    "Qwen3DecoderLayer",
}


def block_dataflow(model, blocks, matrices: list[dict]) -> list[dict]:
    if blocks is None:
        return []
    paths = {id(module): name for name, module in model.named_modules()}
    results = []
    for index, block in enumerate(blocks):
        kind = type(block).__name__
        if kind not in SUPPORTED_BLOCKS:
            results.append({"layer": index, "supported": False, "nodes": [], "edges": []})
            continue
        prefix = paths[id(block)] + "."
        weights = {item["name"]: item for item in matrices if item["name"].startswith(prefix)}
        nodes, edges = [], []

        def node(
            identifier,
            label,
            x,
            y,
            *,
            weight=None,
            detail="",
            weights=weights,
            prefix=prefix,
            nodes=nodes,
        ):
            if weight:
                match = weights.get(prefix + weight + ".weight")
                if not match:
                    raise ValueError(
                        "The block dataflow does not match its logical weight inventory."
                    )
                detail = " × ".join(str(size) for size in match["shape"])
            nodes.append(
                {
                    "id": identifier,
                    "label": label,
                    "x": x,
                    "y": y,
                    "weight": prefix + weight + ".weight" if weight else None,
                    "detail": detail,
                }
            )

        def edge(source, target, label="", *, residual=False, branch=False, edges=edges):
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "label": label,
                    "residual": residual,
                    "branch": branch,
                }
            )

        parallel = kind == "GPTNeoXLayer" and block.use_parallel_residual
        if kind == "GPT2Block":
            qkv, output, up, down = "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"
            norm1, norm2, rotary = "LayerNorm", "LayerNorm", False
        elif kind == "GPTNeoXLayer":
            qkv, output = "attention.query_key_value", "attention.dense"
            up, down = "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"
            norm1, norm2, rotary = "LayerNorm", "LayerNorm", True
        else:
            qkv, output, up, down = None, "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"
            norm1, norm2, rotary = "RMSNorm", "RMSNorm", True
        required = [output, up, down] + (
            [qkv]
            if qkv
            else [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "mlp.gate_proj",
            ]
        )
        if any(prefix + name + ".weight" not in weights for name in required):
            results.append({"layer": index, "supported": False, "nodes": [], "edges": []})
            continue
        node("input", "Residual input", 240, 0)
        node("norm-attn", norm1, 240, 85)
        edge("input", "norm-attn")
        branch_y = 185
        if qkv:
            node("qkv", "Fused QKV projection", 240, 175, weight=qkv)
            edge("norm-attn", "qkv")
            branch_y = 265
            for name, x in (("q", 20), ("k", 240), ("v", 460)):
                node(name, name.upper(), x, branch_y, detail="Split by attention head")
                edge("qkv", name, name.upper())
        else:
            for name, x in (("q", 20), ("k", 240), ("v", 460)):
                node(name, "W_" + name.upper(), x, branch_y, weight="self_attn." + name + "_proj")
                edge("norm-attn", name)
        score_detail = "QKᵀ / √d · causal mask · softmax"
        if rotary:
            score_detail = "RoPE · " + score_detail
        if kind == "Qwen3DecoderLayer":
            score_detail = "Q/K RMSNorm · " + score_detail
        node("scores", "Attention probabilities", 130, branch_y + 105, detail=score_detail)
        edge("q", "scores", "Q")
        edge("k", "scores", "K")
        node(
            "mix",
            "Weighted value sum",
            240,
            branch_y + 215,
            detail="softmax(QKᵀ / √d) V · concatenate heads",
        )
        edge("scores", "mix", "probabilities")
        edge("v", "mix", "V")
        node("out", "W_O", 240, branch_y + 310, weight=output)
        edge("mix", "out")
        mlp_y = branch_y + 510
        if parallel:
            # NeoX feeds both normalized branches from the same incoming residual.
            node("norm-mlp", norm2, 740, 85)
            edge("input", "norm-mlp", "parallel branch", branch=True)
            mlp_x = 740
            mlp_y = 185
            source = "norm-mlp"
        else:
            node("add-attn", "Add attention residual", 240, branch_y + 405)
            edge("out", "add-attn")
            edge("input", "add-attn", "residual", residual=True)
            node("norm-mlp", norm2, 240, mlp_y)
            edge("add-attn", "norm-mlp")
            mlp_x = 240
            mlp_y += 95
            source = "norm-mlp"
        node("up", "W_up" if not qkv else "MLP input", mlp_x, mlp_y, weight=up)
        edge(source, "up")
        activation = type(
            block.mlp.act
            if kind == "GPT2Block"
            else block.mlp.act
            if kind == "GPTNeoXLayer"
            else block.mlp.act_fn
        ).__name__
        if not qkv:
            node("gate", "W_gate", 20, mlp_y, weight="mlp.gate_proj")
            edge(source, "gate")
            node("act", "Activation", 20, mlp_y + 95, detail=activation)
            edge("gate", "act")
            node("multiply", "Elementwise gate × up", 240, mlp_y + 195)
            edge("act", "multiply", "gate")
            edge("up", "multiply", "up")
            mlp_source = "multiply"
            down_y = mlp_y + 290
        else:
            node("act", "Activation", mlp_x, mlp_y + 95, detail=activation)
            edge("up", "act")
            mlp_source, down_y = "act", mlp_y + 195
        node("down", "MLP output", mlp_x, down_y, weight=down)
        edge(mlp_source, "down")
        final_y = max(down_y, branch_y + 310) + 110
        node(
            "output", "Residual output", 240, final_y, detail="Input to the next transformer block"
        )
        edge("down", "output", "MLP contribution")
        if parallel:
            edge("out", "output", "attention contribution")
            edge("input", "output", "residual", residual=True)
        else:
            edge("add-attn", "output", "residual", residual=True)
        results.append(
            {
                "layer": index,
                "supported": True,
                "parallel_residual": parallel,
                "nodes": nodes,
                "edges": edges,
                "scope": "Inference dataflow. Matrix boxes open the actual logical weights. Normalization, position encoding, head reshape and activation operations are summarized. Dropout is disabled in evaluation mode.",
            }
        )
    return results
