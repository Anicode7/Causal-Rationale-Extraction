import json
import os 

with open(os.path.join(os.getcwd(),'data','final_annotated_dataset.json'),'r') as file:
    data = json.load(file)  

for i,node in enumerate(data):
    for j,c in enumerate(node["conversation"]):
        #print(i, j , "\n\n\n")
        #print(node["transcript_id"])

        if "escalation_level" not in c:
            node["conversation"].pop(j)
            continue
        if "escalation_risks" not in c:
            node["conversation"].pop(j)
            continue
        if "churn_risk_score" not in c:     
            node["conversation"].pop(j)
            continue
        if "empathy_score" not in c:
            node["conversation"].pop(j)
            continue



        if not c["escalation_level"]:
            c["escalation_level"] = 0
            print("Escalation level set to 0 for conversation:", c)
        if not c["escalation_risks"]:
            c["escalation_risks"] = 0
            print("Escalation risks set to 0 for conversation:", c)
        if not c["churn_risk_score"]:
            c["churn_risk_score"] = 0
            print("Churn risk score set to 0 for conversation:", c)
        if not c["empathy_score"]:
            c["empathy_score"] = 0
            print("Empathy score set to 0 for conversation:", c)
with open(os.path.join(os.getcwd(),'data','final_annotated_dataset.json'), 'w') as file:
    json.dump(data, file, indent=4, ensure_ascii=False) 