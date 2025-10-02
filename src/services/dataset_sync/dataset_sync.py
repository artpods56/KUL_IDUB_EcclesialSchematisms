import os
import json
import time
import signal
import sys
import logging
import subprocess
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from minio import Minio

from config import setup_logging, load_config_from_env

def run_conversion(config, commit_message_metadata: str):
    """
    Runs the convert_raw_annotations.py script and streams its logs in real-time.
    'config' is the configuration object loaded by dataset_sync.py.
    It must contain 'CONVERT_SCRIPT' (description of the script) and
    'ENV_FILE_PATH_USED_BY_DATASET_SYNC' (path to the .env file dataset_sync.py is using).
    """
    python_executable = sys.executable  # e.g., /usr/local/bin/python
    convert_script_name = config['CONVERT_SCRIPT'] # From config, e.g., "convert_raw_annotations.py"
    
    # Ensure the script path is correct. Assuming it's in /app (the WORKDIR)
    # If CONVERT_SCRIPT is just a description, this should work if /app is in PATH or is current dir.
    # For robustness, you might construct an absolute path if needed: script_full_path = Path("/app") / convert_script_name

    env_file_for_conversion = config.get('ENV_FILE')
    if not env_file_for_conversion:
        logging.error("ENV_FILE not found in config. Cannot run conversion script.")
        raise ValueError("Missing ENV_FILE in configuration for conversion script.")

    command = [
        python_executable,
        convert_script_name,
        '--env_file', str(env_file_for_conversion),
        '--commit_message_metadata', str(commit_message_metadata) # Ensure it's a string
    ]

    logging.info(f"Running conversion script with real-time logging: {' '.join(command)}")

    try:
        # Use Popen to get control over stdout/stderr streams
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr into stdout
            text=True,                # Decode output as text
            bufsize=1,                # Line-buffered
            universal_newlines=True   # Ensure cross-platform newline handling (often implied by text=True)
        )

        # Read and print output line by line
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                # Print the line from the subprocess.
                # The convert_raw_annotations.py script already uses logging,
                # so its output will be formatted log lines.
                # Adding a prefix can help distinguish these logs.
                sys.stdout.write(f"[convert_script] {line.strip()}\n")
                sys.stdout.flush() # Ensure immediate output
            process.stdout.close()

        return_code = process.wait() # Wait for the process to complete

        if return_code == 0:
            logging.info("Conversion script completed successfully.")
        else:
            # Specific error messages from the script should have already been printed.
            error_message = f"Conversion script failed with exit code {return_code}."
            logging.error(error_message)
            # Re-raise an error to indicate failure to the calling sync logic
            raise subprocess.CalledProcessError(return_code, command, output=None, stderr=None)

    except FileNotFoundError:
        logging.error(f"Conversion script '{convert_script_name}' not found. Ensure it is in the correct path.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while running the conversion script: {e}")
        raise
    
def load_state(state_file: str):
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return {}

def save_state(state, state_file: str):
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)


def strip_noncritical(state:dict, key_to_strip: str) -> dict:
    
    stripped_state = {
    k: {subk: subv for subk, subv in v.items() if subk != key_to_strip}
        for k, v in state.items()
    }
    return stripped_state


def get_modified_count(old_state, new_state) -> int:
    new_state = strip_noncritical(new_state, "file")
    old_state = strip_noncritical(old_state, "file")

    modified_count = 0
    for key in old_state.keys() & new_state.keys():  # tylko wspólne
        if old_state[key] != new_state[key]:
            modified_count += 1

    return modified_count



def compute_state_hash(state: dict) -> str:
   
    return sha1(json.dumps(strip_noncritical(state, "file"), sort_keys=True).encode()).hexdigest()

def compute_data_hash(data: dict) -> str:
    return sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()


def list_new_objects(client, minio_bucket):
    objects = {}
    for obj in client.list_objects(minio_bucket, recursive=True):
        objects[obj.object_name] = {
            "size": obj.size,
            "last_modified": str(obj.last_modified)
        }
    return objects

def s3_path_to_identifier(s3_path: Path) -> Path:
    return "_".join(s3_path.parts[-2:]).replace(".jpg", "").replace(".png", "")




def sync_to_directory(client, state: dict, config) -> dict:
    logging.info("Syncing to directory...")
        
    modified_files = {}
    for obj_name, obj_info in state.items():
        annotation_object = client.get_object(
            config['MINIO_BUCKET'], obj_name
        )
        
        annotation_json = json.loads(annotation_object.read().decode('utf-8'))
        local_identifier = s3_path_to_identifier(Path(annotation_json['task']['data']['image']))
        
        json_path = Path(config['LS_ANNOTATIONS_DIR']) / Path(local_identifier + ".json")
        modified_files[obj_name] = local_identifier + ".json"
        
        if not json_path.exists():
            logging.info(f"Saving annotation for {local_identifier} to {json_path}")
            with open(json_path, 'w') as f:
                json.dump(annotation_json, f, indent=4)
        else:
            
            
            
            with open(json_path, 'r') as f:
                existing_annotation = json.load(f)
                if compute_data_hash(annotation_json) != compute_data_hash(existing_annotation):
                    logging.info(f"Updating annotation for {local_identifier} in {json_path}")
                    with open(json_path, 'w') as f:
                        json.dump(annotation_json, f, indent=4)
                    
                    
                else:
                    logging.info(f"Annotation for {local_identifier} already exists, skipping.")

    for key, filename in modified_files.items():
        state[key]["file"] = str(filename)

    return state



def sync_once(client, config):
    old_state = load_state(config['STATE_FILE'])
    old_state_hash = compute_state_hash(old_state)
    
    new_state = list_new_objects(client, config['MINIO_BUCKET'])
    new_state_hash = compute_state_hash(new_state)
    
    if old_state_hash == new_state_hash:
        logging.info("No changes detected in MinIO bucket.")
        return 0
        
    else:
        logging.info("Changes detected in MinIO bucket. Syncing...")
        
        
        
        new_keys = set(new_state.keys()) - set(old_state.keys())
        modified_count = get_modified_count(old_state, new_state)
        
        commit_message_metadata = f"new: {len(new_keys)}, modified: {modified_count}, total: {len(new_state)}"
        
        logging.info(f"New files: {len(new_keys)}, Modified files: {modified_count}")
        
        if modified_count + len(new_keys) >= config['MIN_NEW']:
            logging.info(f"Triggering workflow for {modified_count + len(new_keys)} modified files.")
            
            updated_state = sync_to_directory(client, new_state, config=config)
            run_conversion(config, commit_message_metadata)
            save_state(updated_state, config['STATE_FILE'])






def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Sync MinIO bucket and convert annotations.")
    parser.add_argument('--loop', action='store_true', help="Run in loop mode, syncing every POLL_INTERVAL seconds.")
    parser.add_argument('--env_file', type=str, default='.env', help="Path to the .env file for configuration.")
    return parser.parse_args()


def main():
    
    cli_args = parse_args()
    print(f"Using environment file: {cli_args.env_file}")
    
    config = load_config_from_env(env_file=cli_args.env_file)
    
    setup_logging(config['LOG_LEVEL'])
    config['ENV_FILE'] = cli_args.env_file
    
    
    client = Minio(
        config['MINIO_ENDPOINT'],
        config['MINIO_ACCESS_KEY'],
        config['MINIO_SECRET_KEY'],
        secure=False
    )
    
    
    if cli_args.loop:
        logging.info(f"Running in loop mode. Polling every {config['POLL_INTERVAL']} seconds. Press Ctrl+C to stop.")
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
        
        while True:
            try:
                sync_once(client, config)
                time.sleep(config['POLL_INTERVAL'])
            except KeyboardInterrupt:
                logging.info("Interrupted by user.")
                break
            except Exception as e:
                logging.error(f"Error during sync: {e}")
                time.sleep(config['POLL_INTERVAL'])
    else:
        logging.info("Running in one-time sync mode.")
        sync_once(client, config)

if __name__ == "__main__":
    main()
