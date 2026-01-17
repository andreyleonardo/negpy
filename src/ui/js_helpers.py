from streamlit_javascript import st_javascript  # type: ignore


def get_viewport_height() -> int:
    """
    Returns the browser window inner height in pixels.
    """
    val = st_javascript("window.parent.innerHeight", key="viewport_height_js")
    return int(val) if val else 0
