import json
import pandas as pd
annotated_path = "data/final_annotated_dataset.json"
processed_path = "data/processed_transcripts.json"

with open(annotated_path, "r") as f:
    annotated_data = json.load(f)

with open(processed_path, "r") as f:
    processed_data = json.load(f)

# Convert to DataFrames
df_annotated = pd.DataFrame(annotated_data)
df_processed = pd.DataFrame(processed_data)

# Merge on transcript_id, keeping annotated rows
df_final = df_annotated.merge(
    df_processed[['transcript_id', 'satisfaction_score', 'satisfaction_label']],
    on="transcript_id",
    how="left"
)

print("Final shape:", df_final.shape)
print(df_final.head())

output_path = "data/final_json_for_d2f.json"
df_final.to_json(output_path, orient="records", indent=2)
print(f"Saved enriched dataset to {output_path}")