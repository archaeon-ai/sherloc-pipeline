"""Load the bundled recipe; no runtime fitting or alternate recipe path."""

import hashlib
from importlib.resources import files

from sherloc_pipeline.models.identification import MethodRecipe, MixtureRecipe


def load_recipe() -> tuple[MethodRecipe, str]:
    data = files("sherloc_pipeline.core.identification").joinpath(
        "recipes/raman-pearson-0.2.0.json",
    ).read_bytes()
    return MethodRecipe.model_validate_json(data), hashlib.sha256(data).hexdigest()


def load_mixture_recipe() -> tuple[MixtureRecipe, str]:
    data = files("sherloc_pipeline.core.identification").joinpath(
        "recipes/raman-mixture-0.1.0.json",
    ).read_bytes()
    return MixtureRecipe.model_validate_json(data), hashlib.sha256(data).hexdigest()


def recipe_for_release(release_id):
    for loader in (load_recipe, load_mixture_recipe):
        recipe, identity = loader()
        if identity == release_id:
            return recipe, identity
    raise ValueError("method release is not a bundled frozen recipe")
