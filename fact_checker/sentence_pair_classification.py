import torch

# Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation

claim = "The taking of ibuprofen could be a factor in aggravating the infection"
fact = "Ibuprofen use may be beneficial in COVID-19 disease. "

def fact_check(claim, fact):
    with torch.no_grad():
        tokens = roberta.encode(claim, fact)
        prediction = roberta.predict('mnli', tokens).argmax().item()
    if prediction == 0: 
        check = "False"
    elif prediction == 1:
        check = "Can't tell"
    elif prediction ==2:
         check = "True"   
    return prediction, check
        
prediction, check = fact_check(claim, fact)

print(check)