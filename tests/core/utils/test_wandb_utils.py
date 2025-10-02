import wandb

from core.schemas.data.pipeline import PipelineData
from core.utils.wandb_eval import create_table_from_pydantic

PipelineData


class TestWandbUtils:

    def test_wandb_create_table_from_pydantic(self):

        table = create_table_from_pydantic(PipelineData)

        assert isinstance(table, wandb.Table)
