# Saves models locally for local load
from transformers import AutoTokenizer, AutoModel

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer.save_pretrained("../../models/bio_clinbert_tokenizer")
    model.save_pretrained("../../models/bio_clinbert_model")
