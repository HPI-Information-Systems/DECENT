import os
import subprocess
import wandb


def get_wandb_api_key():
    try:
        api_key = os.environ.get("WANDB_API_KEY")
    except Exception as e:
        print(e)
        api_key = None
    return api_key


def try_wandb_login():
    WAND_API_KEY = get_wandb_api_key()
    if WAND_API_KEY:
        try:
            subprocess.run(["wandb", "login", WAND_API_KEY], check=True)
            return True
        except Exception as e:
            print(e)
            return False
    else:
        print("WARNING: No wandb API key found, this run will NOT be logged to wandb.")
        input("Press any key to continue...")
        return False


def start_wandb_logging(config, project=None, entity=None):
    if project == None:
        project = os.environ.get("WANDB_PROJECT")
    if entity == None:
        entity = os.environ.get("WANDB_ENTITY")
    
    if try_wandb_login():
        run_name = config.output_folder.split('/')[-1]
        wandb.init(project=project, entity=entity, dir=config.output_folder, name=run_name) #, sync_tensorboard=True)

        wandb.config.update(config)

def push_file_to_wandb(filepath):
    wandb.save(filepath)
