


from core.schemas.data.pipeline import PipelineData


class TestPipelineFixtures:
    def test_pipeline_data(self, sample_pipeline_data):
        assert sample_pipeline_data
        assert isinstance(sample_pipeline_data, PipelineData)