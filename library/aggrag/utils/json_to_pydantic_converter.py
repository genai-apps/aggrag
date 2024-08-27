import json
import logging
from pydantic import BaseModel, create_model, ValidationError
from typing import Any, Dict, List, Optional, Union



# Configure logging
logger = logging.getLogger(__name__)

def json_schema_to_pydantic_model(json_schema: Dict[str, Any]) -> BaseModel:
    """
    Convert a JSON schema to a Pydantic model.

    Args:
        json_schema (Dict[str, Any]): The JSON schema to convert.

    Returns:
        BaseModel: A dynamically created Pydantic model.
    """
    try:
        properties = json_schema.get('properties', {})
        required_fields = json_schema.get('required', [])

        # Note: The description should be provided in json schema to return a proper metadata
        model_description = json_schema.get('description', "Dynamically created model")  # Get model description

        fields = {}
        
        for field_name, field_info in properties.items():
            field_type = field_info.get('type')
            
            # Map JSON schema types to Pydantic types
            if field_type == 'string':
                fields[field_name] = (str, ...)
            elif field_type == 'integer':
                fields[field_name] = (int, ...)
            elif field_type == 'number':
                fields[field_name] = (float, ...)
            elif field_type == 'boolean':
                fields[field_name] = (bool, ...)
            elif field_type == 'array':
                items = field_info.get('items', {})
                item_type = items.get('type')
                if item_type == 'string':
                    fields[field_name] = (List[str], ...)
                elif item_type == 'integer':
                    fields[field_name] = (List[int], ...)
                elif item_type == 'number':
                    fields[field_name] = (List[float], ...)
                elif item_type == 'boolean':
                    fields[field_name] = (List[bool], ...)
                elif item_type == 'object':
                    # Handle nested objects
                    nested_model = json_schema_to_pydantic_model(items)
                    fields[field_name] = (List[nested_model], ...)
            elif field_type == 'object':
                # Handle nested objects
                nested_model = json_schema_to_pydantic_model(field_info)
                fields[field_name] = (nested_model, ...)
        
        # Create the Pydantic model dynamically
        DynamicModel = create_model('DynamicModel', **fields, __doc__=model_description)
        
        # Set required fields
        for field in required_fields:
            if field in fields:
                fields[field] = (fields[field][0], ...)
        
        if DynamicModel.model_fields=={}:
            raise Exception("Unable to create a Pydantic Model. Please provide valid metadata_json_schema with required fields, properties and descriptions to create the same.")
        return DynamicModel

    except ValidationError as e:
        logger.error("Validation error occurred while creating Pydantic model: %s", e)
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
        raise




# Example JSON schema provided by the user
# user_provided_schema = '''
# {
#     "type": "object",
#     "properties": {
#         "username": {
#             "type": "string",
#             "description": "The user's unique username."
#         },
#         "email": {
#             "type": "string",
#             "format": "email",
#             "description": "The user's email address."
#         },
#         "age": {
#             "type": "integer",
#             "description": "The user's age."
#         },
#         "tags": {
#             "type": "array",
#             "items": {
#                 "type": "string"
#             },
#             "description": "A list of tags associated with the user."
#         },
#         "address": {
#             "type": "object",
#             "properties": {
#                 "street": {
#                     "type": "string"
#                 },
#                 "city": {
#                     "type": "string"
#                 }
#             },
#             "required": ["street"]
#         }
#     },
#     "required": ["username", "email"]
# }
# '''

# # Load the JSON schema
# schema_dict = json.loads(user_provided_schema)

# # Create the Pydantic model from the JSON schema
# DynamicModel = json_schema_to_pydantic_model(schema_dict)

# # Example of creating an instance of the dynamic model
# instance_data = {
#     "username": "Alice",
#     "email": "alice@example.com",
#     "age": 30,
#     "tags": ["developer", "python"],
#     "address": {
#         "street": "123 Main St",
#         "city": "Wonderland"
#     }
# }

# try:
#     user_instance = DynamicModel(**instance_data)
#     print(user_instance)
# except ValidationError as e:
#     print("Validation error:", e.json())