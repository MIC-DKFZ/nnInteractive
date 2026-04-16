import torch


def cleanse_checkpoint_for_release(checkpoint: str):
    a = torch.load(checkpoint, weights_only=False)
    del a["optimizer_state"]
    del a["init_args"]["dataset_json"]
    a["trainer_name"] = "nnInteractiveTrainer_stub"
    torch.save(a, checkpoint)
