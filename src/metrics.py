import numpy as np

return_entity_level_metrics = False
import numpy as np
import evaluate
# from datasets import load_metric
metric = evaluate.load("seqeval")


def build_compute_metrics(label_list, return_entity_level_metrics=True):
    def compute_metrics(p):
        import numpy as np
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)

        if return_entity_level_metrics:
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        final_results[f"eval_{sub_key.lower()}_{key.capitalize()}"] = sub_val
                else:
                    final_results[f"eval_{key}"] = value
            return final_results
        else:
            return {
                "eval_precision": results["overall_precision"],
                "eval_recall": results["overall_recall"],
                "eval_f1": results["overall_f1"],
                "eval_accuracy": results["overall_accuracy"]
            }
    return compute_metrics
