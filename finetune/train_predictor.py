import os
import sys
import json
import time
from time import gmtime, strftime
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import comet_ml
except ImportError:
    comet_ml = None

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from multihorizon_objective import (
    MultiHorizonForecastHead,
    compute_multihorizon_objective,
)
from model.kronos import KronosTokenizer, Kronos
# Import shared utilities
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time
)


def create_dataloaders(config: dict, rank: int, world_size: int):
    """
    Creates and returns distributed dataloaders for training and validation.

    Args:
        config (dict): A dictionary of configuration parameters.
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset).
    """
    print(f"[Rank {rank}] Creating distributed dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], sampler=val_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=True, drop_last=False
    )
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(
    model,
    tokenizer,
    forecast_head,
    device,
    config,
    save_dir,
    logger,
    rank,
    world_size,
):
    """
    The main training and validation loop for the predictor.
    """
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, Total: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

    trainable_parameters = list(model.parameters()) + list(forecast_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['predictor_learning_rate'],
        steps_per_epoch=len(train_loader), epochs=config['epochs'],
        pct_start=0.03, div_factor=10
    )

    best_val_loss = float('inf')
    stale_epochs = 0
    dt_result = {}
    batch_idx_global = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        forecast_head.train()
        train_loader.sampler.set_epoch(epoch_idx)

        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)

        for i, (batch_x, batch_x_stamp, batch_raw_close) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            batch_raw_close = batch_raw_close.to(device, non_blocking=True)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass and loss calculation
            s1_logits, s2_logits, hidden_states = model(
                token_in[0],
                token_in[1],
                batch_x_stamp[:, :-1, :],
                return_context=True,
            )
            model_core = model.module if hasattr(model, "module") else model
            token_loss, s1_loss, s2_loss = model_core.head.compute_loss(
                s1_logits,
                s2_logits,
                token_out[0],
                token_out[1],
            )
            forecast_result = compute_multihorizon_objective(
                forecast_head,
                hidden_states,
                batch_raw_close,
                context_length=config['lookback_window'],
                horizons=config['forecast_horizons'],
                min_deadzone=config['direction_min_deadzone'],
                volatility_multiplier=config['direction_volatility_multiplier'],
                huber_delta=config['return_huber_delta'],
            )
            return_loss = forecast_result['return_loss']
            direction_loss = forecast_result['direction_loss']
            loss = (
                token_loss
                + config['return_loss_weight'] * return_loss
                + config['direction_loss_weight'] * direction_loss
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=3.0)
            optimizer.step()
            scheduler.step()

            # Logging (Master Process Only)
            if rank == 0 and (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Loss: {loss.item():.4f}"
                )
            if rank == 0 and logger:
                lr = optimizer.param_groups[0]['lr']
                logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                logger.log_metric('train_token_loss_batch', token_loss.item(), step=batch_idx_global)
                logger.log_metric('train_return_loss_batch', return_loss.item(), step=batch_idx_global)
                logger.log_metric('train_direction_loss_batch', direction_loss.item(), step=batch_idx_global)
                logger.log_metric('train_direction_accuracy_batch', forecast_result['direction_accuracy'].item(), step=batch_idx_global)
                for horizon, accuracy in zip(
                    config['forecast_horizons'],
                    forecast_result['direction_accuracy_by_horizon'],
                ):
                    logger.log_metric(
                        f'train_direction_accuracy_h{horizon}_batch',
                        accuracy.item(),
                        step=batch_idx_global,
                    )
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
        forecast_head.eval()
        tot_val_loss_sum_rank = 0.0
        tot_val_token_loss_sum_rank = 0.0
        tot_val_return_loss_sum_rank = 0.0
        tot_val_direction_loss_sum_rank = 0.0
        tot_val_direction_correct_by_horizon_rank = torch.zeros(
            len(config['forecast_horizons']), device=device
        )
        tot_val_direction_points_rank = 0
        val_batches_processed_rank = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp, batch_raw_close in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                batch_raw_close = batch_raw_close.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                s1_logits, s2_logits, hidden_states = model(
                    token_in[0],
                    token_in[1],
                    batch_x_stamp[:, :-1, :],
                    return_context=True,
                )
                model_core = model.module if hasattr(model, "module") else model
                val_token_loss, _, _ = model_core.head.compute_loss(
                    s1_logits,
                    s2_logits,
                    token_out[0],
                    token_out[1],
                )
                forecast_result = compute_multihorizon_objective(
                    forecast_head,
                    hidden_states,
                    batch_raw_close,
                    context_length=config['lookback_window'],
                    horizons=config['forecast_horizons'],
                    min_deadzone=config['direction_min_deadzone'],
                    volatility_multiplier=config['direction_volatility_multiplier'],
                    huber_delta=config['return_huber_delta'],
                )
                val_return_loss = forecast_result['return_loss']
                val_direction_loss = forecast_result['direction_loss']
                val_loss = (
                    val_token_loss
                    + config['return_loss_weight'] * val_return_loss
                    + config['direction_loss_weight'] * val_direction_loss
                )

                tot_val_loss_sum_rank += val_loss.item()
                tot_val_token_loss_sum_rank += val_token_loss.item()
                tot_val_return_loss_sum_rank += val_return_loss.item()
                tot_val_direction_loss_sum_rank += val_direction_loss.item()
                tot_val_direction_correct_by_horizon_rank += (
                    forecast_result['direction_accuracy_by_horizon'].detach()
                    * batch_x.shape[0]
                )
                tot_val_direction_points_rank += batch_x.shape[0]
                val_batches_processed_rank += 1

        # Reduce validation metrics
        val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
        val_batches_tensor = torch.tensor(val_batches_processed_rank, device=device)
        val_token_loss_tensor = torch.tensor(tot_val_token_loss_sum_rank, device=device)
        val_return_loss_tensor = torch.tensor(tot_val_return_loss_sum_rank, device=device)
        val_direction_loss_tensor = torch.tensor(tot_val_direction_loss_sum_rank, device=device)
        val_direction_points_tensor = torch.tensor(tot_val_direction_points_rank, device=device)
        if dist.is_initialized():
            dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_token_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_return_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_direction_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(tot_val_direction_correct_by_horizon_rank, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_direction_points_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = val_loss_sum_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0
        avg_val_token_loss = val_token_loss_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0
        avg_val_return_loss = val_return_loss_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0
        avg_val_direction_loss = val_direction_loss_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0
        avg_val_direction_accuracy_by_horizon = (
            tot_val_direction_correct_by_horizon_rank / val_direction_points_tensor
            if val_direction_points_tensor.item() > 0
            else torch.zeros(len(config['forecast_horizons']), device=device)
        )
        avg_val_direction_accuracy = avg_val_direction_accuracy_by_horizon.mean().item()

        # --- End of Epoch Summary & Checkpointing (Master Process Only) ---
        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(
                f"Validation Token Loss: {avg_val_token_loss:.4f}, "
                f"Return Loss: {avg_val_return_loss:.4f}, "
                f"Direction Loss: {avg_val_direction_loss:.4f}, "
                f"Direction Accuracy: {avg_val_direction_accuracy:.2%}"
            )
            print(
                "Validation Direction by Horizon: "
                + ", ".join(
                    f"h{horizon}={accuracy.item():.2%}"
                    for horizon, accuracy in zip(
                        config['forecast_horizons'],
                        avg_val_direction_accuracy_by_horizon,
                    )
                )
            )
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)
                logger.log_metric('val_token_loss_epoch', avg_val_token_loss, epoch=epoch_idx)
                logger.log_metric('val_return_loss_epoch', avg_val_return_loss, epoch=epoch_idx)
                logger.log_metric('val_direction_loss_epoch', avg_val_direction_loss, epoch=epoch_idx)
                logger.log_metric('val_direction_accuracy_epoch', avg_val_direction_accuracy, epoch=epoch_idx)
                for horizon, accuracy in zip(
                    config['forecast_horizons'], avg_val_direction_accuracy_by_horizon
                ):
                    logger.log_metric(
                        f'val_direction_accuracy_h{horizon}_epoch',
                        accuracy.item(),
                        epoch=epoch_idx,
                    )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                stale_epochs = 0
                save_path = f"{save_dir}/checkpoints/best_model"
                model_core = model.module if hasattr(model, "module") else model
                forecast_core = forecast_head.module if hasattr(forecast_head, "module") else forecast_head
                model_core.save_pretrained(save_path)
                torch.save(
                    {
                        'state_dict': forecast_core.state_dict(),
                        'd_model': model_core.d_model,
                        'head_type': 'multi_horizon_return_and_direction',
                        'horizons': list(config['forecast_horizons']),
                        'pool_size': config['forecast_pool_size'],
                        'lookback_window': config['lookback_window'],
                        'direction_classes': ['down', 'flat', 'up'],
                        'direction_min_deadzone': config['direction_min_deadzone'],
                        'direction_volatility_multiplier': config['direction_volatility_multiplier'],
                    },
                    os.path.join(save_path, 'multihorizon_head.pt'),
                )
                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
            else:
                stale_epochs += 1

        should_stop = (
            config['early_stopping_patience'] > 0
            and stale_epochs >= config['early_stopping_patience']
        ) if rank == 0 else False
        if dist.is_initialized():
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            if rank == 0:
                print(
                    "Early stopping: validation loss did not improve for "
                    f"{config['early_stopping_patience']} consecutive epoch(s)."
                )
            break

        if dist.is_initialized():
            dist.barrier()

    dt_result['best_val_loss'] = best_val_loss
    dt_result['forecast_horizons'] = list(config['forecast_horizons'])
    dt_result['return_loss_weight'] = config['return_loss_weight']
    dt_result['direction_loss_weight'] = config['direction_loss_weight']
    return dt_result


def main(config: dict):
    """Main function to orchestrate the DDP training process."""
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])

    # Logger and summary setup (master process only)
    comet_logger, master_summary = None, {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
        }
        if config['use_comet']:
            if comet_ml is None:
                raise RuntimeError(
                    "Comet logging is enabled but comet_ml is not installed"
                )
            comet_logger = comet_ml.Experiment(
                api_key=config['comet_config']['api_key'],
                project_name=config['comet_config']['project_name'],
                workspace=config['comet_config']['workspace'],
            )
            comet_logger.add_tag(config['comet_tag'])
            comet_logger.set_name(config['comet_name'])
            comet_logger.log_parameters(config)
            print("Comet Logger Initialized.")

    if dist.is_initialized():
        dist.barrier()

    # Model Initialization
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval().to(device)

    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    model.to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    model_core = model.module if hasattr(model, "module") else model
    forecast_head = MultiHorizonForecastHead(
        model_core.d_model,
        horizons=config['forecast_horizons'],
        pool_size=config['forecast_pool_size'],
        dropout=config['forecast_head_dropout'],
    ).to(device)
    if dist.is_initialized():
        forecast_head = DDP(
            forecast_head,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    if rank == 0:
        print(f"Predictor Model Size: {get_model_size(model_core)}")

    # Start Training
    dt_result = train_model(
        model,
        tokenizer,
        forecast_head,
        device,
        config,
        save_dir,
        comet_logger,
        rank,
        world_size,
    )

    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if comet_logger: comet_logger.end()

    cleanup_ddp()


if __name__ == '__main__':
    # Usage: torchrun --standalone --nproc_per_node=NUM_GPUS train_predictor.py
    if "WORLD_SIZE" not in os.environ and os.environ.get("KRONOS_SINGLE_PROCESS") != "1":
        raise RuntimeError("This script must be launched with `torchrun`.")

    config_instance = Config()
    main(config_instance.__dict__)
