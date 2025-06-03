from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D

def load_labels(dataset: Dataset):
    classes = []
    for example in dataset:
        if "labels" in example.keys():
            if isinstance(example["labels"], list):
                classes.extend(example["labels"])
            else:
                classes.append(example["labels"])
                
    unique_classes = set(classes)
    sorted_classes = sorted(list(unique_classes))

    id2label = {i: label for i, label in enumerate(sorted_classes)}
    label2id = {label: i for i, label in enumerate(sorted_classes)}
    return id2label, label2id, sorted_classes


def prepare_dataset(dataset: Dataset, processor, id2label, label2id, dataset_config):
    
    def prepare_examples(examples):
        images = examples[dataset_config["image_column_name"]]
        words = examples[dataset_config["text_column_name"]]
        boxes = examples[dataset_config["boxes_column_name"]]
        word_labels = examples[dataset_config["label_column_name"]]

        # Since your dataset has string labels, always convert them to IDs
        label_ids = [
            [label2id[label] for label in seq]
            for seq in word_labels
        ]

        encoding = processor(images, words, boxes=boxes, word_labels=label_ids, truncation=True, stride =128, 
                padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True)  
        offset_mapping = encoding.pop('offset_mapping')
        overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')
        return encoding
    
    


    features = Features({
        'input_ids': Sequence(Value("int64")),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'attention_mask': Sequence(Value("int64")),
        'labels': Sequence(Value("int64")),
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    })

    prepared_dataset = dataset.map(
        prepare_examples,
        batched=True,
        remove_columns=dataset.column_names,
        features=features,
    )
    
    prepared_dataset.set_format("torch")
    
    return prepared_dataset

