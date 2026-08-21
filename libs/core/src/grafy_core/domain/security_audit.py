from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4


class SecurityAuditActorKind(StrEnum):
    AUTHENTICATED = "authenticated"
    UNAUTHENTICATED = "unauthenticated"
    SYSTEM = "system"


class SecurityAuditOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class SecurityAuditEvent:
    actor_kind: SecurityAuditActorKind
    operation: str
    outcome: SecurityAuditOutcome
    id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=_utc_now)
    user_id: UUID | None = None
    credential_reference: str | None = None
    workspace_id: UUID | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        self.actor_kind = SecurityAuditActorKind(self.actor_kind)
        self.outcome = SecurityAuditOutcome(self.outcome)
        if self.operation.strip() == "" or len(self.operation) > 120:
            raise ValueError("Security audit operation must be 1-120 characters")
        if self.resource_type is not None and len(self.resource_type) > 80:
            raise ValueError(
                "Security audit resource type must be at most 80 characters"
            )
        if self.resource_id is not None and len(self.resource_id) > 255:
            raise ValueError(
                "Security audit resource id must be at most 255 characters"
            )
        if (
            self.credential_reference is not None
            and len(self.credential_reference) > 120
        ):
            raise ValueError(
                "Security audit credential reference must be at most 120 characters"
            )
        if self.error_code is not None and len(self.error_code) > 80:
            raise ValueError("Security audit error code must be at most 80 characters")
        if self.actor_kind is SecurityAuditActorKind.AUTHENTICATED:
            if self.user_id is None:
                raise ValueError("Authenticated audit events require a user id")
        elif self.user_id is not None or self.credential_reference is not None:
            raise ValueError(
                "Unauthenticated and system audit events cannot carry user or "
                "credential attribution"
            )
        if self.outcome is SecurityAuditOutcome.SUCCESS and self.error_code is not None:
            raise ValueError("Successful audit events cannot carry an error code")
        if self.outcome is SecurityAuditOutcome.FAILURE and self.error_code is None:
            raise ValueError("Failed audit events require a safe error code")
        if self.occurred_at.tzinfo is None:
            raise ValueError("Security audit timestamp must be timezone-aware")


__all__ = [
    "SecurityAuditActorKind",
    "SecurityAuditEvent",
    "SecurityAuditOutcome",
]
