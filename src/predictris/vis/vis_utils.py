def confidence_score_to_color(confidence: float) -> str:
    """Convert confidence score to color hex code (red -> yellow -> green gradient)."""
    if confidence <= 0.5:
        # Red to yellow (interpolate between #00FF00 and #FFFF00)
        ratio = confidence * 2  # Scale 0-0.5 to 0-1
        green = int(255 * ratio)
        red = 255
    else:
        # Yellow to green (interpolate between #FFFF00 and #FF0000)
        ratio = (confidence - 0.5) * 2  # Scale 0.5-1 to 0-1
        green = 255
        red = int(255 * (1 - ratio))
    
    return f'#{red:02x}{green:02x}00'