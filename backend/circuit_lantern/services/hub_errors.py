"""Explain Hub failures without exposing remote responses or credentials."""

import httpx
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import (
    GatedRepoError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


def architecture_diagnostic(error: Exception, config_directory: str) -> str:
    message = str(error).replace(config_directory, "the cached checkpoint")
    message = message.replace(str(HF_HUB_CACHE), "the model cache")
    if len(message) > 16384:
        message = message[:16384].rsplit(" ", 1)[0] + "\nFurther diagnostics were omitted."
    return message


def explain_hub_error(error: Exception) -> tuple[int, str, dict[str, str]]:
    response = getattr(error, "response", None)
    status = response.status_code if response is not None else 503
    headers = {}
    if isinstance(error, GatedRepoError):
        status = 403
        message = (
            "This model requires access approval on Hugging Face. Accept its terms or request "
            "access on the model page, then set HF_TOKEN on the server for an approved account."
        )
    elif isinstance(error, RevisionNotFoundError):
        status = 404
        message = (
            "This model revision does not exist on Hugging Face. Check the branch, tag or "
            "commit, or choose the model's default revision."
        )
    elif isinstance(error, RemoteEntryNotFoundError):
        status = 422
        try:
            path = response.request.url.path if response is not None else ""
        except RuntimeError:
            path = ""
        if path.endswith("/config.json"):
            message = (
                "This repository does not contain config.json. Choose a full Transformers "
                "safetensors checkpoint. GGUF, MLX and adapter-only repositories cannot be "
                "loaded by this runtime."
            )
        else:
            message = (
                "A required checkpoint file is missing from this model revision. Choose a "
                "complete Transformers safetensors checkpoint or another revision."
            )
    elif isinstance(error, RepositoryNotFoundError):
        status = 404
        message = (
            "This model repository was not found or is private. Check its repository ID. "
            "For a private model, set HF_TOKEN on the server with read access to that repository."
        )
    elif status == 429:
        message = (
            "Hugging Face has temporarily limited requests from this server. Try again later. "
            "A server HF_TOKEN can increase the Hub request allowance."
        )
    elif status in (401, 403):
        message = (
            "Hugging Face denied access to this request. Check that the server HF_TOKEN is "
            "valid and has read access to the model. Gated models also require approval."
        )
    elif status == 404:
        message = (
            "Hugging Face could not find the requested model or file. Check its ID and revision."
        )
    elif isinstance(error, OfflineModeIsEnabled):
        status = 503
        message = (
            "Hugging Face access is disabled by server offline mode. Disable HF_HUB_OFFLINE "
            "to search models and resolve checkpoint revisions online."
        )
    elif isinstance(error, httpx.TimeoutException):
        status = 504
        message = "Hugging Face did not respond in time. Try again shortly."
    elif isinstance(error, (httpx.RequestError, LocalEntryNotFoundError)):
        status = 503
        message = (
            "The server could not reach Hugging Face and the requested file is not available "
            "locally. Check the server connection or proxy, then try again."
        )
    else:
        status = 503 if status >= 500 else 502
        message = "Hugging Face is temporarily unable to provide this model. Try again shortly."
    if status in (429, 503, 504):
        retry = response.headers.get("Retry-After", "") if response is not None else ""
        headers["Retry-After"] = (
            retry
            if len(retry) <= 4 and retry.isascii() and retry.isdecimal() and 0 < int(retry) <= 3600
            else "2"
        )
    return status, message, headers
