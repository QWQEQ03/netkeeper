"""Versioned, schema-only evaluation framework for NetKeeper methods."""

from netkeeper_sim.evaluation.methods import (
    EvaluationContext, EvaluationMethod, MethodDecision, MethodMetadata,
    DispatcherMethodAdapter, UnavailableMethod,
)
from netkeeper_sim.evaluation.manifest import EVALUATION_MANIFEST_VERSION, generate_evaluation_manifest
from netkeeper_sim.evaluation.runner import EvaluationRunner, RunOutcome
from netkeeper_sim.evaluation.results import EVALUATOR_VERSION, ResultStore, configuration_change
from netkeeper_sim.evaluation.aggregate import aggregate_runs
from netkeeper_sim.evaluation.baselines import LocalSearchOSPFMethod, NoUpdateMethod, OSPFDefaultMethod, RandomMethod

__all__ = [
    "DispatcherMethodAdapter", "EVALUATION_MANIFEST_VERSION", "EVALUATOR_VERSION",
    "EvaluationContext", "EvaluationMethod", "EvaluationRunner", "MethodDecision",
    "MethodMetadata", "ResultStore", "RunOutcome", "UnavailableMethod",
    "aggregate_runs", "configuration_change", "generate_evaluation_manifest",
    "LocalSearchOSPFMethod", "NoUpdateMethod", "OSPFDefaultMethod", "RandomMethod",
]
