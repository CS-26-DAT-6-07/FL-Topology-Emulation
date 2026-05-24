import torch
import os
import sklearn
import json
import matplotlib.pyplot as plt


from models.xception import xception
from dataset.dataset import no_augment_init_dataset,load_centralized_dataset


def evaluate(batch,model, device):
    images = batch["image"].to(device)
    act = batch["label"]
    with torch.no_grad():
        pred = model(images.float())
    pred_labels = torch.argmax(pred, dim=1)
    
    return act.cpu().tolist(), pred_labels.cpu().tolist()

model_name = "fedavg_seed36836"
generate_report = True

model = xception()
model.load_state_dict(torch.load(os.getcwd()+f"/finished_models/{model_name}.pt",weights_only=True))
model.float()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

no_augment_init_dataset()
dataloader = load_centralized_dataset()

act_y = []
pred_y = []
count = 1
for batch in dataloader:
    print(f"batch {count}")
    act_list, pred_list = evaluate(batch, model=model, device=device)
    act_y += act_list
    pred_y += pred_list
    count += 1

#confusion_matrix = sklearn.metrics.confusion_matrix(act_y,pred_y,labels=[i for i in range(0,8)], normalize='pred')

if generate_report:
    report = sklearn.metrics.classification_report(act_y,pred_y,labels=[i for i in range(0,8)], output_dict=True)

    print(report)

    write_string = json.dumps(report)


    with open(f"{model_name}_report.json","w") as f:
        f.write(write_string)

sklearn.metrics.ConfusionMatrixDisplay.from_predictions(act_y,pred_y,labels=[i for i in range(0,8)], normalize='pred')
plt.show()