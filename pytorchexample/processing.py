import torch
import os
import sklearn
import json

from models.xception import xception
from dataset.dataset import no_augment_init_dataset,load_centralized_dataset


def evaluate(batch,model, device):
    images = batch["image"].to(device)
    act = batch["label"]
    with torch.no_grad():
        pred = model(images)
    pred_labels = torch.argmax(pred, dim=1)
    
    return act.cpu().tolist(), pred_labels.cpu().tolist()

model = xception()
model.load_state_dict(torch.load(os.getcwd()+"/Dataprocessing/fedavg_cycle_model.pt",weights_only=True))
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

report = sklearn.metrics.classification_report(act_y,pred_y,labels=[i for i in range(0,8)], output_dict=True)

print(report)

write_string = json.dumps(report)


with open("report.json","w") as f:
    f.write(write_string)


