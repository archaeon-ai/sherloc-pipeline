"""Load the bundled recipe; no runtime fitting or alternate recipe path."""

import hashlib
from importlib.resources import files

from sherloc_pipeline.models.identification import MethodRecipe


def load_recipe() -> tuple[MethodRecipe, str]:
    data = files("sherloc_pipeline.core.identification").joinpath(
        "recipes/raman-pearson-0.2.0.json",
    ).read_bytes()
    return MethodRecipe.model_validate_json(data), hashlib.sha256(data).hexdigest()


def recipe_for_release(release_id):
    recipe, identity = load_recipe()
    if identity == release_id:
        return recipe, identity
    raise ValueError("method release is not a bundled frozen recipe")
