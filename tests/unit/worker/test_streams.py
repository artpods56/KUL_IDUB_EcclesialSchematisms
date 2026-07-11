from nats.js.api import RetentionPolicy, StorageType, StreamConfig
from notarius_messaging.subjects import (
    ARTIFACT_DELTA_SUBJECT_WILDCARD,
    ARTIFACT_EVENT_SUBJECT_WILDCARD,
    EVENT_SUBJECTS,
    LIVE_DELTA_SUBJECTS,
    NODE_RUN_DELTA_SUBJECT_WILDCARD,
    NODE_RUN_EVENT_SUBJECT_WILDCARD,
    NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
    TASK_SUBJECTS,
    WORKFLOW_COMPILE_REQUESTED_SUBJECT,
    WORKFLOW_RUN_DELTA_SUBJECT_WILDCARD,
    WORKFLOW_RUN_EVENT_SUBJECT_WILDCARD,
    WORKFLOW_RUN_REQUESTED_SUBJECT,
)
from notarius_worker.streams import (
    CONSUMER_NODE_RUN_EXECUTE,
    create_worker_streams,
    node_run_consumer_config,
    stream_config_for,
    worker_stream_config_matches,
    worker_stream_update_config,
)


def test_task_stream_contains_workflow_and_node_run_subjects() -> None:
    streams = create_worker_streams()

    config = stream_config_for(streams.tasks)

    assert config.name == "TASKS"
    assert config.subjects == list(TASK_SUBJECTS)
    assert config.subjects == [
        WORKFLOW_COMPILE_REQUESTED_SUBJECT,
        WORKFLOW_RUN_REQUESTED_SUBJECT,
        NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
    ]
    assert config.retention == RetentionPolicy.WORK_QUEUE


def test_event_and_live_delta_subjects_are_separate_streams() -> None:
    streams = create_worker_streams()

    events_config = stream_config_for(streams.events)
    live_config = stream_config_for(streams.live_deltas)

    assert events_config.subjects == list(EVENT_SUBJECTS)
    assert events_config.subjects == [
        WORKFLOW_RUN_EVENT_SUBJECT_WILDCARD,
        NODE_RUN_EVENT_SUBJECT_WILDCARD,
        ARTIFACT_EVENT_SUBJECT_WILDCARD,
    ]
    assert live_config.subjects == list(LIVE_DELTA_SUBJECTS)
    assert live_config.subjects == [
        WORKFLOW_RUN_DELTA_SUBJECT_WILDCARD,
        NODE_RUN_DELTA_SUBJECT_WILDCARD,
        ARTIFACT_DELTA_SUBJECT_WILDCARD,
    ]
    assert live_config.storage == StorageType.MEMORY
    assert events_config.storage == StorageType.FILE


def test_node_run_consumer_uses_long_running_execution_backoff() -> None:
    config = node_run_consumer_config(
        CONSUMER_NODE_RUN_EXECUTE,
        NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
    )

    assert config.filter_subject == NODE_RUN_EXECUTE_REQUESTED_SUBJECT
    assert config.ack_wait == 600
    assert config.backoff == [600, 1800, 3600, 7200, 7200]
    assert config.backoff[0] == config.ack_wait


def test_worker_stream_update_config_reconciles_managed_fields() -> None:
    existing = StreamConfig(
        name="TASKS",
        subjects=["jobs.workflow.run.requested"],
        retention=RetentionPolicy.LIMITS,
        storage=StorageType.MEMORY,
        max_age=10,
        duplicate_window=1,
    )
    desired = StreamConfig(
        name="TASKS",
        subjects=list(TASK_SUBJECTS),
        retention=RetentionPolicy.WORK_QUEUE,
        storage=StorageType.FILE,
        max_age=604800,
        duplicate_window=120,
    )

    updated = worker_stream_update_config(existing, desired)

    assert updated.subjects == desired.subjects
    assert updated.retention == desired.retention
    assert updated.storage == desired.storage
    assert updated.max_age == desired.max_age
    assert updated.duplicate_window == desired.duplicate_window


def test_worker_stream_config_matches_compares_managed_fields() -> None:
    config = StreamConfig(
        name="TASKS",
        subjects=["jobs.node_run.execute.requested"],
        retention=RetentionPolicy.WORK_QUEUE,
        storage=StorageType.FILE,
        max_age=604800,
        duplicate_window=120,
    )

    assert worker_stream_config_matches(config, config)
    assert not worker_stream_config_matches(
        config,
        StreamConfig(
            name="TASKS",
            subjects=["jobs.workflow.run.requested"],
            retention=RetentionPolicy.WORK_QUEUE,
            storage=StorageType.FILE,
            max_age=604800,
            duplicate_window=120,
        ),
    )


def test_subject_vocabulary_is_workflow_node_run_and_artifact_centered() -> None:
    all_subjects = " ".join((*TASK_SUBJECTS, *EVENT_SUBJECTS, *LIVE_DELTA_SUBJECTS))

    assert "workflow" in all_subjects
    assert "node_run" in all_subjects
    assert "artifact" in all_subjects
    subject_tokens = {
        token
        for subject in (*TASK_SUBJECTS, *EVENT_SUBJECTS, *LIVE_DELTA_SUBJECTS)
        for token in subject.split(".")
    }
    assert "recipe" not in subject_tokens
    assert "job" not in subject_tokens
