class ConfigFileNotFoundError(Exception):
    """Raise when specified config file is not found."""
    def __init__(self, config_name, config_type, available_configs):
        message = (
            f"Config file for {config_type} '{config_name}' not found.",
            f"Available configs are: {available_configs}"
        )
        super().__init__(message)
        self.config_name = config_name
        self.config_type = config_type
        self.available_configs = available_configs


class ConfigNotRegisteredError(Exception):
    """Raise when specified config is not registered."""
    def __init__(self, config_type, config_subtype, registered_configs):
        message = (
            f"Config '{config_subtype}' for {config_type} is not registered.",
            f"Please register a schema using @register_config('{config_type.value}', '{config_subtype.value}')."
            f"Currently registered configs: {registered_configs}"
        )
        super().__init__(message)
        self.config_type = config_type
        self.config_subtype = config_subtype
        self.registered_configs = registered_configs

class InvalidConfigType(Exception):
    """Raise when specified config type is invalid."""
    def __init__(self, config_type, available_config_types):
        message = (
            f"Invalid config type: {config_type.value}",
            f"Available types: {available_config_types}"
        )
        super().__init__(message)
        self.config_type = config_type
        self.available_types = available_config_types

class InvalidConfigSubtype(Exception):
    """Raise when specified config subtype is invalid."""
    def __init__(self, config_type, config_subtype, supported_subtypes):
        message = (
            f"Invalid config subtype: {config_subtype} for {config_type}",
            f"This config type supports the following subtypes: {supported_subtypes}"
        )
        super().__init__(message)
        self.config_type = config_type
        self.config_subtype = config_subtype
        self.supported_subtypes = supported_subtypes

class ConfigValidationError(Exception):
    """Raise when config validation fails."""
    def __init__(self, message):
        super().__init__(message)