from synthecg.render.layouts.layout_12x1 import render_layout_12x1
from synthecg.render.layouts.layout_3x4 import render_layout_3x4

LAYOUT_RENDERERS = {
    "3x4+1": render_layout_3x4,
    "12x1": render_layout_12x1,
}


def render_layout(record, config, *, ecg_id, patient_id, scp_codes):
    renderer = LAYOUT_RENDERERS.get(config.layout)
    if renderer is None:
        raise ValueError(f"Unknown layout '{config.layout}'. Available: {list(LAYOUT_RENDERERS)}")
    return renderer(record, config, ecg_id=ecg_id, patient_id=patient_id, scp_codes=scp_codes)
