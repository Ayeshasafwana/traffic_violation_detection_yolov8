def check_violation(detections, signal="RED"):
    violations = []

    # Custom YOLO class IDs
    motorcycle_detected = 1 in detections
    vehicle_detected = any(
        cls in detections for cls in [0, 1, 2, 3, 4, 5]
    )

    # Helmet detection is not available in this model.
    # Do not claim actual no-helmet detection.

    # Signal logic (prototype simulation)
    if vehicle_detected and signal == "RED":
        violations.append("Possible Signal Jump")

    if not violations:
        return ["No Violation"]

    return violations