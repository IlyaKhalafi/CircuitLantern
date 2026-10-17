"""Verified mixer semantics for dense Qwen 3.5 text blocks.

The surrounding block and MLP connections are derived from installed forward
source. These adapters describe kernels that static module inspection cannot
expand, and only activate for the installed, known implementation classes.
"""


def expand_hybrid_mixer(graph, module, path, inputs):
    if not type(module).__module__.startswith("transformers.models.qwen3_5."):
        return None
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5Attention,
            Qwen3_5GatedDeltaNet,
        )
    except ImportError:
        return None

    if type(module) not in (Qwen3_5Attention, Qwen3_5GatedDeltaNet):
        return None
    delta = type(module) is Qwen3_5GatedDeltaNet
    required = (
        ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
        if delta
        else ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    if any(path + "." + name + ".weight" not in graph.weights for name in required):
        return None
    start = len(graph.nodes)

    def op(label, refs, detail=""):
        return graph.node(label, refs, detail=detail, section="attention")

    def weight(name, refs, label):
        full = path + "." + name + ".weight"
        return graph.node(
            label,
            refs,
            weight=full,
            section="attention",
            detail=" × ".join(map(str, graph.weights[full]["shape"])),
        )

    if delta:
        qkv = weight("in_proj_qkv", inputs, "Fused QKV projection")
        conv = op("Causal convolution", qkv, "Depthwise convolution · SiLU · uncached sequence")
        query = op("Q", conv, "Split QKV · reshape heads · L2 normalization in kernel")
        key = op("K", conv, "Split QKV · reshape heads · L2 normalization in kernel")
        value = op("V", conv, "Split QKV · reshape value heads")
        beta = op("Sigmoid", weight("in_proj_b", inputs, "Update projection"))
        decay = op(
            "Decay gate",
            weight("in_proj_a", inputs, "Decay projection"),
            "−exp(A_log) × softplus(a + dt_bias)",
        )
        core = op(
            "Gated DeltaNet recurrence",
            query + key + value + beta + decay,
            "Chunked gated delta rule · recurrent state · not softmax attention",
        )
        z = weight("in_proj_z", inputs, "Output gate projection")
        gated = op("Gated RMSNorm", core + z, "RMSNorm(core) × SiLU(z)")
        output = weight("out_proj", gated, "Mixer output projection")
    else:
        qgate = weight("q_proj", inputs, "Query and gate projection")
        query = op("Q", qgate, "Split query · Q RMSNorm · RoPE")
        gate = op("Sigmoid output gate", qgate, "Split gate · sigmoid")
        key = op("K", weight("k_proj", inputs, "W_K"), "K RMSNorm · RoPE")
        value = weight("v_proj", inputs, "W_V")
        probabilities = op(
            "Attention probabilities", query + key, "QKᵀ / √d · causal mask · softmax"
        )
        mix = op(
            "Weighted value sum",
            probabilities + value,
            "Attention probabilities × V · concatenate heads",
        )
        gated = op("Multiply output gate", mix + gate, "Attention output × sigmoid(gate)")
        output = weight("o_proj", gated, "W_O")
    for node in graph.nodes[start:]:
        node["section"] = "attention"
    graph.mixer_type = "linear_attention" if delta else "full_attention"
    return output
