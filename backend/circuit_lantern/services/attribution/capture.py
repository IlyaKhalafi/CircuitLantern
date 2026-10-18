"""Observe actual block boundaries without retaining full prompt activations."""

import logging

import torch

from circuit_lantern.services.attribution.normalization import UnsupportedAttribution

MAX_CAPTURE_BYTES = 64 * 1024 * 1024


def last_vector(value):
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
        raise UnsupportedAttribution("capture_shape")
    return value[0, -1].detach().clone()


def input_hidden(args, kwargs):
    return args[0] if args else kwargs.get("hidden_states", kwargs.get("input"))


class AttributionCapture:
    """Hooks apply only to the prompt pass and are always removed before generation."""

    def __init__(self, blocks, norm, head, config):
        self.blocks, self.norm, self.head, self.config = blocks, norm, head, config
        self.handles = []
        self.before, self.after = {}, {}
        self.norm_input = self.norm_output = self.head_input = None
        self.reason = None
        self.dtype = None

    def install(self):
        if self.reason:
            return
        if self.blocks is None or not len(self.blocks):
            self.reason = "missing_blocks"
            return
        if (
            not isinstance(self.head, torch.nn.Linear)
            or type(self.head).forward is not torch.nn.Linear.forward
        ):
            self.reason = "unsupported_head"
            return
        if not self.head.weight.is_floating_point() or type(self.head.weight) not in (
            torch.Tensor,
            torch.nn.Parameter,
        ):
            self.reason = "quantized_head"
            return
        if (len(self.blocks) * 2 + 4) * self.head.in_features * 8 > MAX_CAPTURE_BYTES:
            self.reason = "capture_budget"
            return
        if len({id(block) for block in self.blocks}) != len(self.blocks):
            self.reason = "reused_blocks"
            return
        try:
            for index, block in enumerate(self.blocks):

                def before(module, args, kwargs, index=index):
                    self.observe(self.before, index, input_hidden(args, kwargs))

                def after(module, args, output, index=index):
                    self.observe(
                        self.after, index, output[0] if isinstance(output, tuple) else output
                    )

                self.handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
                self.handles.append(block.register_forward_hook(after))
            if self.norm is not None:
                self.handles.append(
                    self.norm.register_forward_pre_hook(self.before_norm, with_kwargs=True)
                )
                self.handles.append(self.norm.register_forward_hook(self.after_norm))
            self.handles.append(
                self.head.register_forward_pre_hook(self.before_head, with_kwargs=True)
            )
        except Exception:
            self.remove()
            self.clear()
            raise

    def observe(self, target, index, value):
        if self.reason:
            return
        try:
            if index in target:
                raise UnsupportedAttribution("reused_blocks")
            vector = last_vector(value)
            target[index] = vector
            self.dtype = vector.dtype
        except UnsupportedAttribution as error:
            self.reason = str(error)
            self.clear()

    def before_norm(self, module, args, kwargs):
        if self.reason:
            return
        try:
            if self.norm_input is not None:
                raise UnsupportedAttribution("reused_norm")
            self.norm_input = last_vector(input_hidden(args, kwargs))
        except UnsupportedAttribution as error:
            self.reason = str(error)

    def after_norm(self, module, args, output):
        if self.reason:
            return
        try:
            self.norm_output = last_vector(output)
        except UnsupportedAttribution as error:
            self.reason = str(error)

    def before_head(self, module, args, kwargs):
        if self.reason:
            return
        try:
            if self.head_input is not None:
                raise UnsupportedAttribution("reused_head")
            self.head_input = last_vector(input_hidden(args, kwargs))
        except UnsupportedAttribution as error:
            self.reason = str(error)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self):
        self.before.clear()
        self.after.clear()
        self.norm_input = self.norm_output = self.head_input = None

    def summarize(self, logits, tokenizer, targets):
        from circuit_lantern.services.attribution.projection import decompose

        try:
            if self.reason:
                raise UnsupportedAttribution(self.reason)
            return decompose(self, logits, tokenizer, targets)
        except UnsupportedAttribution as error:
            return {
                "schema_version": 1,
                "status": "unavailable",
                "reason": str(error),
                "position": -1,
            }
        except Exception:
            logging.getLogger(__name__).exception("Optional logit attribution failed")
            return {
                "schema_version": 1,
                "status": "unavailable",
                "reason": "diagnostic_error",
                "position": -1,
            }
        finally:
            self.clear()
