from types import NoneType, UnionType
from typing import Type, Callable, Union, Dict, Any, Literal, get_args, get_origin
from pydantic import BaseModel
from pathlib import Path

import rich_click as click


def _strip_optional_type(field_type: type | str | None):
    """
    Extracts the non-`None` type from an `Optional` / `Union` type.

    Args:
        field_type: Type of field for Axolotl CLI command.

    Returns:
        If the input type is `Union[T, None]` or `Optional[T]`, returns `T`. Otherwise
            returns the input type unchanged.
    """
    if (get_origin(field_type) is Union or isinstance(field_type, UnionType)) and type(None) in get_args(field_type):
        field_type = next(
            t for t in get_args(field_type) if not isinstance(t, NoneType)
        )

    return field_type

def _is_basic_instance(x: Any):
    return isinstance(x, bool) or isinstance(x, str) or isinstance(x, int) or isinstance(x, float)

def _is_basic_type(x: type):
    return (x is bool) or (x is str) or (x is int) or (x is float)

def _determine_click_type(field_type: Any):
    if _is_basic_type(field_type):
        return field_type
    elif field_type is Path:
        return click.Path(path_type=Path)
    elif get_origin(field_type) is Literal:
        args = get_args(field_type)
        if all([_is_basic_instance(x) for x in args]):
            return click.Choice(args)
    else:
        return None


def add_options_from_config(config_class: Type[BaseModel], force: bool = False) -> Callable:
    """
    Create Click options from the fields of a Pydantic model.
    """

    def decorator(function: Callable) -> Callable:
        # Process model fields in reverse order for correct option ordering
        for name, field in reversed(config_class.model_fields.items()):
            
            extra_schema = field.json_schema_extra if field.json_schema_extra else {}
            if (not force) and (not extra_schema.get('is_cli', False)):
                continue

            option_names = []
            field_type = _strip_optional_type(field.annotation)
            if field_type is bool:
                field_name = name.replace("_", "-")
                option_name = f"--{field_name}/--no-{field_name}"
            else:
                option_name = option_name = f"--{name.replace('_', '-')}"
                
            short_option = extra_schema.get('cli_short', None)
            option_names = [option_name, short_option] if short_option else [option_name]

            function = click.option(
                *option_names, default=None, help=field.description,
                type=_determine_click_type(field_type)
            )(function)

        return function

    return decorator
