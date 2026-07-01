from synthecg.config import AugmentConfig, GenerationConfig, RenderConfig
from synthecg.recipes.definitions import get_recipe


def config_from_recipe(
    recipe_name: str,
    *,
    output_dir: str,
    seed: int | None = None,
    resume: bool = False,
    workers: int | None = None,
) -> GenerationConfig:
    """Build a GenerationConfig from a named recipe."""
    recipe = get_recipe(recipe_name)

    render_data = recipe.pop("render", {})
    augment_profile = recipe.pop("augment_profile", "scan")
    recipe.pop("description", None)

    render = RenderConfig(**render_data)
    augment = AugmentConfig(profile=augment_profile, seed=seed)

    if workers is not None:
        recipe["workers"] = workers

    config = GenerationConfig(
        output_dir=output_dir,
        seed=seed,
        resume=resume,
        render=render,
        augment=augment,
        **recipe,
    )
    return config
