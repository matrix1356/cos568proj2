from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

model_dir = "/scratch/gpfs/zs8839/cos5682/cos568proj2/bert_model"

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
config = AutoConfig.from_pretrained("bert-base-cased")

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
config.save_pretrained(model_dir)