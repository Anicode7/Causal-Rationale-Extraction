import os
import json
import logging
from pathlib import Path

# Import your existing modules
import query_splitter
import llm_handler
# Import the GraphGenerator class from your provided file
from graph_gen import GraphGenerator  
import graph_operator

# Setup Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def save_transcripts_for_pipeline(conversations, output_dir):
    """
    Takes the set of conversation objects from query_splitter and saves them 
    in the text format that GraphGenerator Step 2 expects.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving {len(conversations)} transcripts to {output_dir}...")

    for conv in conversations:

        t_id = conv.get('transcript_id', 'unknown_id')
        domain = conv.get('domain', 'general')
        intent = conv.get('intent', 'unknown')
        reason = conv.get('reason_for_call', 'N/A')
        
        turns = conv.get('conversation', []) 
        if not turns and 'turns' in conv:
            turns = conv['turns']

        filename = os.path.join(output_dir, f"{t_id}.txt")
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Write Header Metadata (Essential for Step 2 parsing)
            f.write(f"Transcript ID: {t_id}\n")
            f.write(f"Domain: {domain}\n")
            f.write(f"Intent: {intent}\n")
            f.write(f"Reason: {reason}\n")
            # These scores might not exist since we skipped search, defaulting to 1.0
            f.write(f"Max Similarity: 1.0000\n") 
            f.write(f"Avg Similarity: 1.0000\n")
            f.write("\nTop Matching Turns:\n")
            f.write("=" * 60 + "\n")
            
            # Write the actual dialogue
            for turn in turns:
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                f.write(f"\n{speaker}: {text}\n")
                f.write(f"Similarity: 1.0000\n") # Dummy score needed for regex parsers often

def run_pipeline_from_transcripts(
    conversations, 
    data_path, 
    distance_threshold=0.5,
    n_clusters=None,
    model_name="sergioburdisso/dialog2flow-joint-bert-base"
):
    """
    Orchestrates the pipeline skipping Step 1.
    """
    pipeline = GraphGenerator(data_path=data_path)
    pipeline.clean_directories()
    
    if not conversations:
        logger.error("No conversations provided to generate graph.")
        return
    save_transcripts_for_pipeline(conversations, pipeline.top20_dir)
    try:
        # Step 2: Prepare for Dialog2Flow
        results_meta = pipeline.step2_prepare_for_dialog2flow()
        
        # Step 3: Extract Trajectories
        results_d2f = pipeline.step3_extract_dialog2flow_trajectories(
            model_name=model_name,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold
        )
        
        # Step 4: Attach Metadata
        results_enriched = pipeline.step4_attach_metadata_to_trajectories()
        
        # Step 5: Build Graphs
        results_graphs = pipeline.step5_build_graphs(
            formats=['json', 'graphml', 'html']
        )
        
        logger.info(f"Graph generation complete! Output in: {pipeline.output_dir}")
        return pipeline
        
    except Exception as e:
        logger.error(f"Pipeline failed during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    QUERY = "How do fare increases and decreases impact customer satisfaction, and what are the optimal fare management strategies?"
    DATA_PATH = 'data/final_annotated_dataset.json' # Path to your main dataset
    
    llm = llm_handler.llm() 
    
    print(f"--- 1. Splitting Query & Retrieving Transcripts ---")
    retrieved_conversations = query_splitter.categorize_query(QUERY, llm)
    
    print(f"--- 2. Generating Graph from {len(retrieved_conversations)} Transcripts ---")
    
    if len(retrieved_conversations) > 0:
        run_pipeline_from_transcripts(
            conversations=retrieved_conversations,
            data_path=DATA_PATH,
            distance_threshold=0.6
        )
        ans, _ = graph_operator.get_ans(QUERY)
        print(f"Answer generated: ", ans)
    else:
        print("No transcripts found. Exiting.")