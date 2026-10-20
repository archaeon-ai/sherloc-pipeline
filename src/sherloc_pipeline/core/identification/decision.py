"""Fixed provisional abstention; request-time threshold fitting is absent."""

from sherloc_pipeline.models.identification import MethodRecipe


def decide(ranking, quality: dict, recipe: MethodRecipe) -> tuple[str, list[str]]:
    reason = quality.get("reason")
    if reason:
        return "abstained", [reason]
    if quality.get("snr_proxy", 0) < recipe.snr_min:
        return "abstained", ["insufficient_signal"]
    if not ranking:
        return "abstained", ["no_compatible_species_references"]
    if ranking[0].support < recipe.support_min:
        return "abstained", ["below_support_threshold"]
    return "candidate", []
