from fastapi import APIRouter

from notarius_api.v1.routes import (
    artifacts,
    artifact_types,
    experiments,
    exports,
    jobs,
    node_specs,
    node_runs,
    outbox,
    projects,
    prototype,
    recipes,
    schemas,
    sources,
    workflow_templates,
    workflows,
)

router = APIRouter()
router.include_router(projects.router)
router.include_router(sources.router)
router.include_router(schemas.router)
router.include_router(recipes.router)
router.include_router(jobs.router)
router.include_router(exports.router)
router.include_router(workflow_templates.router)
router.include_router(workflows.router)
router.include_router(experiments.router)
router.include_router(node_specs.router)
router.include_router(prototype.router)
router.include_router(artifact_types.router)
router.include_router(node_runs.router)
router.include_router(outbox.router)
router.include_router(artifacts.router)
