# Notarius Studio Platform Documentation

Status: draft platform direction

This folder records the current platform decisions for Notarius Studio. The goal is to give future implementation agents and maintainers a stable description of what the platform is, which concepts matter, and how the staged backend refactor should proceed.

## Product Summary

Notarius Studio is a reproducible artifact-graph workbench for assembling, running, comparing, and auditing AI/ML pipelines over sequential research sources.

The platform should support OCR, layout analysis, vision models, LLM-based extraction, classical ML models, evaluation, export, and future model families without making any one model type the center of the architecture.

The core primitive is not the visual node editor. The core primitive is the typed artifact graph:

```text
Operator execution produces versioned artifacts.
Artifacts have declared types, schemas, lineage, metadata, and inspectable payloads.
Workflows are reproducible compositions of those artifacts.
```

## Reading Order

1. [Product Direction](product-direction.md)
   Explains the product framing, target users, MVP slice, and what Notarius Studio is not.

2. [User Experience](user-experience.md)
   Describes the webapp from a researcher perspective: projects, templates, canvas, run view, artifact inspection, page timeline, and comparison views.

3. [Conceptual Model](conceptual-model.md)
   Defines the main platform objects and justifies why each one exists.

4. [Artifact Graph](artifact-graph.md)
   Describes artifacts, artifact sequences, lineage, payloads, and provenance.

5. [MVP Source Ingestion Slice](mvp-source-ingestion-slice.md)
   Describes the first mixed-source workflow shape: connector-backed source nodes, image sequence merging, and resolver-based downstream consumption.

6. [Node And Execution Model](node-and-execution-model.md)
   Explains operator specs, runtime node runs, execution modes, workflow compilation, and traces.

7. [Operator Catalog](operator-catalog.md)
   Lists the initial operator families, including source import, OCR, layout, input assembly, model invocation, evaluation, and export.

8. [Example Workflows](example-workflows.md)
   Shows concrete platform workflows such as four-page OCR comparison, contextual structured extraction, experiment matrices, DAG resolution, and JetStream execution.

9. [Backend Architecture](backend-architecture.md)
   Describes the target package layout, API, worker, persistence, storage, NATS JetStream, and reusable code from the Stacking backend.

10. [Implementation Stages](implementation-stages.md)
   Gives a staged implementation plan for coding agents.

11. [Stacking Backend Reuse Map](stacking-backend-reuse-map.md)
   Lists which parts of the Stacking backend should be adapted and which parts should not be copied.

## Vocabulary

- **Artifact**: A typed, versioned, evidence-bearing result or input object.
- **Artifact graph**: The directed graph of artifacts and the operator runs that produced them.
- **Workflow definition**: The editable graph saved by the user.
- **Workflow version**: An immutable snapshot of a workflow definition.
- **Workflow run**: One execution of a workflow version against inputs.
- **Operator spec**: The registered contract for a node type. The UI may call this a node.
- **Node run**: One concrete execution of an operator inside a workflow run.
- **Input assembly trace**: The record of how inputs were selected, resolved, pruned, or bundled before execution.
- **Invocation trace**: The record of an external or local model/tool invocation.

## Design Constraint

Do not specialize the core around LLM extraction. LLM prompts, responses, token usage, and chat history are important, but they belong to LLM operator payloads and invocation traces. The core platform should be equally comfortable with OCR engines, layout detectors, embedding models, classifiers, segmentation models, and structured extraction models.
