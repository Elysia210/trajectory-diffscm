from pathlib import Path
import os

import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    config.dataset_name = "TRAJECTORY"
    config.experiment_name = "trajectory_diff_scm_baseline"

    use_gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpus

    config.data = data = ml_collections.ConfigDict()
    data.path = Path("/mnt/h/trajectory_apr11/Apr11_relaxed_all_archives")
    data.expected_timesteps = 100
    data.recursive = True
    data.cache_in_memory = False
    data.require_labels = False
    data.label_map_path = None
    data.label_candidates = (
        "collision",
        "is_collision",
        "collided",
        "collision_label",
        "label",
        "target",
        "y",
        "no_collision",
        "is_no_collision",
    )
    data.train_ratio = 0.8
    data.val_ratio = 0.2
    data.test_ratio = 0.0
    data.num_workers = 0

    # Keep the same top-level structure used by existing configs even though
    # the first-stage trajectory prototype only needs the classifier path.
    config.diffusion = diffusion = ml_collections.ConfigDict()
    diffusion.steps = 1000
    diffusion.learn_sigma = False
    diffusion.sigma_small = False
    diffusion.noise_schedule = "linear"
    diffusion.use_kl = False
    diffusion.rescale_learned_sigmas = False
    diffusion.predict_xstart = False
    diffusion.rescale_timesteps = False
    diffusion.timestep_respacing = "ddim100"
    diffusion.conditioning_noise = "constant"

    config.score_model = score_model = ml_collections.ConfigDict()
    score_model.class_cond = False
    score_model.classifier_free_cond = False
    score_model.image_level_cond = False

    # First trajectory-domain diffusion prototype. It conditions on the first
    # half of the pair trajectory and denoises/generates the future half.
    config.trajectory_diffusion = trajectory_diffusion = ml_collections.ConfigDict()
    trajectory_diffusion.input_dim = 23
    trajectory_diffusion.history_steps = 50
    trajectory_diffusion.hidden_dim = 128
    trajectory_diffusion.num_layers = 2
    trajectory_diffusion.dropout = 0.1
    trajectory_diffusion.time_embed_dim = 64
    # Training-side acceleration/smoothness regularizer weight on the predicted
    # step sequence (0 disables it; set >0 to make the model's own samples smoother).
    trajectory_diffusion.accel_reg_weight = 0.0

    config.trajectory_diffusion.training = diffusion_training = ml_collections.ConfigDict()
    diffusion_training.epochs = 20
    diffusion_training.batch_size = 64
    diffusion_training.lr = 1e-3
    diffusion_training.weight_decay = 1e-4
    diffusion_training.grad_clip = 1.0

    config.classifier = classifier = ml_collections.ConfigDict()
    classifier.label = ["collision"]
    classifier.input_dim = 23
    classifier.hidden_dim = 128
    classifier.num_layers = 2
    classifier.dropout = 0.1
    classifier.bidirectional = True

    config.classifier.training = training = ml_collections.ConfigDict()
    training.iterations = 20
    training.batch_size = 64
    training.lr = 1e-3
    training.weight_decay = 1e-4
    training.grad_clip = 1.0
    training.log_interval = 1
    training.save_interval = 1
    training.resume_checkpoint = ""
    training.threshold = 0.5
    training.pos_weight = None

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.batch_size = training.batch_size

    config.seed = 42
    config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return config
