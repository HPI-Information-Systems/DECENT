from yacs.config import CfgNode as ConfigurationNode

# YACS overwrite these settings using YAML
__C = ConfigurationNode()


# WANDB
__C.wandb = ConfigurationNode(new_allowed=True)
# Disable wandb logging
__C.wandb.offline = False
__C.wandb.entity = None
__C.wandb.project = None
# Use to name wandb run
__C.wandb.name = None


# MODEL
__C.model = ConfigurationNode(new_allowed=True)
# Huggingface transformer model name
__C.model.model_name = "roberta-large"
# Enable cross attention in classification head
__C.model.head_cross_attention = True
# Enable self attention in classification head
__C.model.head_self_attention = True
# Use mean of label tokens for representation, otherwise use [CLS] token 
__C.model.use_label_mean = True
# Dropout probability of classification head
__C.model.head_dropout_prob = 0.2
# Use a models weights from existing checkpoint
__C.model.from_ckpt = None
# Start threshold for best model search
__C.model.threshold_start = None
# Step size for best model search; if 'None' best model is not searched
__C.model.threshold_step = None
# Margin for similarity loss; if defined, model uses cosine similarity instead of classification head
__C.model.sim_loss_margin = None
# Freeze encoder to be not trainable
__C.model.freeze_encoder = False
# If defined and no prediction exceeds threshold, use this label (int) as fallback; otherwise label with highest probability is chosen
__C.model.label_fallback = None
# Optimizer parameters
__C.model.optimizer_params = ConfigurationNode(new_allowed=True)
# Scheduler parameters
__C.model.scheduler_params = ConfigurationNode(new_allowed=True)

# DATASET
__C.data = ConfigurationNode(new_allowed=True)
# Paths to data
__C.data.data_files = ConfigurationNode(new_allowed=True)
__C.data.data_files.train = None
__C.data.data_files.val = None
__C.data.data_files.test = None
# Path to labels
__C.data.label_path = None
# Path to train labels, only used if train labels differ from evaluation labels
__C.data.train_label_path = None
# Batch size for training
__C.data.train_batch_size = None
# Batch size for validation and testing
__C.data.eval_batch_size = None
# Choose number of workers for data loaders, if 'None' automatically choose based on the environment
__C.data.num_workers = None
# Restrict tokenized max length of data for faster results
__C.data.max_length = 128
# If 'None' use whole dataset, otherwise randomly select 'subset' samples; usually used for debugging
__C.data.subset = None
# Randomly sample negative examples for each positive example
__C.data.neg_batch_factor = 1
# Restrict tokenized max length of labels for faster results
__C.data.label_max_length = 32
# If 'True' use the mention marker token as representation token, otherwise [CLS] token
__C.data.use_enclosing_token_pos = True


# PYTORCH LIGHTNING TRAINER
__C.trainer = ConfigurationNode(new_allowed=True)
__C.trainer.max_epochs = None
__C.trainer.accelerator = "auto"
__C.trainer.gpus = None
__C.trainer.gradient_clip_val = 1
__C.trainer.num_sanity_val_steps = 0
__C.trainer.precision = 16
__C.trainer.val_check_interval = None
__C.trainer.default_root_dir = None


def get_default_config():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return __C.clone()
