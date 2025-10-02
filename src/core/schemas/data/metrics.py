from pydantic import BaseModel


class Metrics(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        val = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        return round(val, 3)

    @property
    def recall(self) -> float:
        val = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return round(val, 3)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        val = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(val, 3)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn
        val = self.tp / total if total else 1.0
        return round(val, 3)

    def update(self, tp: int = 0, fp: int = 0, fn: int = 0):
        self.tp += tp
        self.fp += fp
        self.fn += fn


class PageDataMetrics(BaseModel):
    page_number: Metrics
    parish: Metrics
    deanery: Metrics
    dedication: Metrics
    building_material: Metrics
