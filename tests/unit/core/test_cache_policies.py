from notarius_core.operators.arithmetic import ARITHMETIC
from notarius_core.operators.images import IMAGES
from notarius_core.operators.modules import MODULES
from notarius_core.operators.prompts import PROMPTS
from notarius_core.operators.schemas import SCHEMAS
from notarius_core.operators.sequences import SEQUENCES
from notarius_core.operators.text import TEXT
from notarius_core.plugins import NodeCachePolicy


def test_builtin_node_cache_policy_inventory_is_fail_closed() -> None:
    policies = {
        registration.key: registration.cache_policy
        for plugin in (
            IMAGES,
            MODULES,
            SEQUENCES,
            ARITHMETIC,
            TEXT,
            SCHEMAS,
            PROMPTS,
        )
        for registration in plugin.nodes
    }

    assert policies == {
        ("image.upload", 1): NodeCachePolicy.NEVER,
        ("module.input", 1): NodeCachePolicy.NEVER,
        ("module.output", 1): NodeCachePolicy.EXACT,
        ("sequence.collect", 1): NodeCachePolicy.EXACT,
        ("sequence.count", 1): NodeCachePolicy.EXACT,
        ("sequence.slice", 1): NodeCachePolicy.EXACT,
        ("sequence.item_at", 1): NodeCachePolicy.EXACT,
        ("arithmetic.number", 1): NodeCachePolicy.EXACT,
        ("arithmetic.integer_sequence", 1): NodeCachePolicy.EXACT,
        ("arithmetic.add", 1): NodeCachePolicy.EXACT,
        ("arithmetic.subtract", 1): NodeCachePolicy.EXACT,
        ("arithmetic.multiply", 1): NodeCachePolicy.EXACT,
        ("arithmetic.sum", 1): NodeCachePolicy.EXACT,
        ("text.input", 1): NodeCachePolicy.EXACT,
        ("text.split", 1): NodeCachePolicy.EXACT,
        ("text.replace", 1): NodeCachePolicy.EXACT,
        ("text.join", 1): NodeCachePolicy.EXACT,
        ("schema.builder", 1): NodeCachePolicy.EXACT,
        ("prompt.message.create", 2): NodeCachePolicy.EXACT,
    }
