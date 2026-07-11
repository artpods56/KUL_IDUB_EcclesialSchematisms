"""Service for merging datasets with OCR text."""

from notarius_schematisms.domain.dataset import (
    BaseItemDataset,
    GroundTruthItemDataset,
    PredictionDataItem,
    PredictionItemDataset,
)


class MergingService:
    """Service for merging datasets with OCR text using LEFT JOIN on image_path."""

    def merge_predictions_with_ocr(
        self,
        predictions: PredictionItemDataset,
        ocr: BaseItemDataset,
    ) -> PredictionItemDataset:
        """Merge prediction dataset with OCR text.

        Args:
            predictions: Dataset with prediction items
            ocr: Dataset with OCR text

        Returns:
            PredictionItemDataset with OCR text injected
        """
        ocr_map = self._build_ocr_map(ocr)
        merged_items: list[PredictionDataItem] = []

        for item in predictions.items:
            if not item.image_path:
                continue

            merged_items.append(
                PredictionDataItem(
                    image_path=item.image_path,
                    metadata=item.metadata,
                    predictions=item.predictions,
                    text=ocr_map.get(item.image_path),
                )
            )

        return PredictionItemDataset(items=merged_items)

    def merge_ground_truth_with_ocr(
        self,
        ground_truth: GroundTruthItemDataset,
        ocr: BaseItemDataset,
    ) -> PredictionItemDataset:
        """Merge ground truth dataset with OCR, converting to prediction format.

        Ground truth entries are placed in the predictions field to allow
        downstream processing (e.g., source generation) to work uniformly.

        Args:
            ground_truth: Dataset with ground truth items
            ocr: Dataset with OCR text

        Returns:
            PredictionItemDataset with ground truth as predictions and OCR text
        """
        ocr_map = self._build_ocr_map(ocr)
        merged_items: list[PredictionDataItem] = []

        for item in ground_truth.items:
            if not item.image_path:
                continue

            merged_items.append(
                PredictionDataItem(
                    image_path=item.image_path,
                    metadata=item.metadata,
                    predictions=item.ground_truth,  # Convert ground_truth → predictions
                    text=ocr_map.get(item.image_path),
                )
            )

        return PredictionItemDataset(items=merged_items)

    def _build_ocr_map(self, ocr: BaseItemDataset) -> dict[str, str | None]:
        """Build lookup map from image path to OCR text."""
        return {
            item.image_path: item.text
            for item in ocr.items
            if item.image_path
        }
