import numpy as np
import pandas as pd
import torch
import argparse
import evaluate
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer


def main():

    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, help='filename of test data')
    args = parser.parse_args()    
    model_name = args.model_name
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    
    # ----------------
    # functions
    # ----------------
    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
        inputs['labels'] = example_batch['labels']
        return inputs
    
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }
    
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        return metric.compute(
            predictions=np.argmax(eval_pred.predictions, axis=1), 
            references=eval_pred.label_ids
        )
    
    # ----------------
    # iterate each split
    # ----------------    
    for s in range(5):
        print('======================')
        print(f'-- Split: {s}')
        
        ## prepare data
        print('-- Load Data...')
        dataset = load_dataset("imagefolder", data_dir=f"planet-imgs-original/split{s+1}/")
        prepared_ds = dataset.with_transform(transform)
        
        ## prepare labels
        labels = [0,1]
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label
        
        ## load model
        print('-- Load Model...')
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=2,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes = True
        )
        
        ## training args
        output_path = model_name.split("/")[-1]+f'/split{s+1}'
        training_args = TrainingArguments(
            output_dir=output_path,
            per_device_train_batch_size=2,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",    
            eval_steps=20,
            save_steps=20,
            logging_steps=20,
            num_train_epochs=10,
            fp16=True,
            learning_rate=5e-6,
            save_total_limit=1,
            remove_unused_columns=False,
            push_to_hub=False,
            load_best_model_at_end=True,
        )        
        
        ## trainer
        print('-- Training...')
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=prepared_ds["train"],
            eval_dataset=prepared_ds["validation"],
            tokenizer=feature_extractor,
        )        
        
        ## training
        train_results = trainer.train()
        
        ## evaluation
        print('-- Validating...')
        val = trainer.evaluate(prepared_ds['validation'])
        test = trainer.evaluate(prepared_ds['test'])
        trainer.log_metrics(f"validation_{s}", val)
        trainer.save_metrics(f"validation_{s}", val)
        trainer.log_metrics(f"test_{s}", test)
        trainer.save_metrics(f"test_{s}", test)        
        
if __name__ == "__main__":
    main()