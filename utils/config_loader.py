import yaml
import os

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    verification = config.setdefault("verification", {})
    bool_value = os.getenv("VERIFICATION_ENABLED")
    if bool_value is not None:
        normalized = bool_value.strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError("VERIFICATION_ENABLED must be true or false")
        verification["enabled"] = normalized == "true"
    int_settings = {
        "MAX_EVIDENCE_SOURCES": ("max_evidence_sources", 1, 3),
        "MAX_VERIFICATION_RETRIES": ("max_verification_retries", 1, 3),
    }
    for env_name, (setting_name, minimum, maximum) in int_settings.items():
        raw_value = os.getenv(env_name)
        if raw_value is not None:
            value = int(raw_value)
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{env_name} must be between {minimum} and {maximum}"
                )
            verification[setting_name] = value
    raw_timeout = os.getenv("VERIFICATION_EVIDENCE_TIMEOUT_SECONDS")
    if raw_timeout is not None:
        timeout = float(raw_timeout)
        if not 0.1 <= timeout <= 15:
            raise ValueError(
                "VERIFICATION_EVIDENCE_TIMEOUT_SECONDS must be between 0.1 and 15"
            )
        verification["evidence_timeout_seconds"] = timeout
    return config
