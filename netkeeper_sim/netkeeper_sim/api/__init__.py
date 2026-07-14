"""Safe, versioned API definition and validation boundary."""
from netkeeper_sim.api.models import ApiCall, ApiError, ApiRequest, ApiResponse, ExecutionPlan, OptimizationRequest, ValidationResult
from netkeeper_sim.api.registry import API_REGISTRY, API_VERSION, api_definitions, export_json_schema, export_response_json_schema
from netkeeper_sim.api.validator import RANGES, validate_request
from netkeeper_sim.api.executor import OptimizerDispatcher, execute

__all__ = ["ApiCall", "ApiError", "ApiRequest", "ApiResponse", "ExecutionPlan", "OptimizationRequest", "ValidationResult", "API_REGISTRY", "API_VERSION", "RANGES", "api_definitions", "export_json_schema", "export_response_json_schema", "validate_request", "OptimizerDispatcher", "execute"]
