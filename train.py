from __future__ import absolute_import, division, print_function
# coding=utf-8

import os
import time
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader 

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def calculate_verification_metrics(similarities, labels):
    """Calculate verification metrics including EER, ROC AUC, etc."""
    from sklearn.metrics import roc_auc_score, roc_curve
    import numpy as np
    
    # Convert to numpy arrays
    similarities = similarities.cpu().numpy() if torch.is_tensor(similarities) else similarities
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    
    # Calculate EER (Equal Error Rate)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    # Calculate ROC AUC
    auc = roc_auc_score(labels, similarities)
    
    return {
        'eer': eer,
        'auc': auc,
        'eer_threshold': eer_threshold,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }

def verification_loss(emb1, emb2, labels, margin=0.4):
    """
    Compute verification loss using contrastive learning approach
    Args:
        emb1, emb2: embeddings of image pairs
        labels: 1 for same person, 0 for different person
        margin: margin for contrastive loss
    """
    # Normalize embeddings
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    
    # Calculate cosine similarity
    cos_sim = torch.sum(emb1 * emb2, dim=1)
    
    # Contrastive loss: minimize distance for same pairs, maximize for different pairs
    pos_loss = (1 - cos_sim) * labels  # For same person (label=1), minimize distance
    neg_loss = torch.clamp(cos_sim - margin, min=0) * (1 - labels)  # For different person (label=0), maximize distance
    
    loss = pos_loss + neg_loss
    return loss.mean(), cos_sim

def save_model(args, model, optimizer=None, scheduler=None, global_step=0, best_acc=0):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    checkpoint = {
        'model': model_to_save.state_dict(),
        'global_step': global_step,
        'best_acc': best_acc,
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s] at step %d", args.output_dir, global_step)

def load_checkpoint(args, model, optimizer=None, scheduler=None):
    """Load checkpoint for resuming training"""
    checkpoint_path = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    if not os.path.exists(checkpoint_path):
        logger.info("No checkpoint found at %s, starting from scratch", checkpoint_path)
        return 0, 0  # global_step, best_acc
    
    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Load model state
    model.load_state_dict(checkpoint['model'])
    
    # Load training state
    global_step = checkpoint.get('global_step', 0)
    best_acc = checkpoint.get('best_acc', 0)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Loaded optimizer state")
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded scheduler state")
    
    logger.info("Resumed from step %d with best accuracy %f", global_step, best_acc)
    return global_step, best_acc

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step    

    if args.dataset == "NDTWIN":
        num_classes = 2  # Binary classification: same person (1) or different person (0)

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)

    model.load_from(np.load(args.pretrained_dir))
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train_verification(args, model):
    """ Train the model for verification task """
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Mixed precision training
    scaler = None
    if args.fp16:
        scaler = torch.amp.GradScaler('cuda')

    # Train!
    logger.info("***** Running verification training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    
    # Load checkpoint if resuming training
    global_step, best_acc = load_checkpoint(args, model, optimizer, scheduler)
    start_time = time.time()
    
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            img1, img2, labels = batch

            # Get embeddings for both images
            emb1 = model(img1, return_embedding=True)
            emb2 = model(img2, return_embedding=True)
            
            # Calculate verification loss and similarity
            loss, cos_sim = verification_loss(emb1, emb2, labels, margin=args.margin)
            
            # Convert similarity to binary prediction using threshold
            preds = (cos_sim > args.similarity_threshold).long()

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(labels.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], labels.detach().cpu().numpy(), axis=0
                )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0:
                    with torch.no_grad():
                        accuracy = valid_verification(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model, optimizer, scheduler, global_step, accuracy)
                        best_acc = accuracy
                    logger.info("best accuracy so far: %f" % best_acc)
                    model.train()

                if global_step % t_total == 0:
                    break
        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        logger.info("train accuracy so far: %f" % accuracy)
        losses.reset()
        if global_step % t_total == 0:
            break

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))

def valid_verification(args, model, writer, test_loader, global_step):
    # Validation for verification task
    eval_losses = AverageMeter()

    logger.info("***** Running Verification Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_similarities, all_labels = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        img1, img2, labels = batch
        with torch.no_grad():
            # Get embeddings for both images
            emb1 = model(img1, return_embedding=True)
            emb2 = model(img2, return_embedding=True)
            
            # Calculate verification loss and similarity
            eval_loss, cos_sim = verification_loss(emb1, emb2, labels, margin=args.margin)
            eval_losses.update(eval_loss.item())

        if len(all_similarities) == 0:
            all_similarities.append(cos_sim.detach().cpu())
            all_labels.append(labels.detach().cpu())
        else:
            all_similarities[0] = torch.cat([all_similarities[0], cos_sim.detach().cpu()], dim=0)
            all_labels[0] = torch.cat([all_labels[0], labels.detach().cpu()], dim=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_similarities, all_labels = all_similarities[0], all_labels[0]
    
    # Calculate verification metrics
    metrics = calculate_verification_metrics(all_similarities, all_labels)
    
    # Calculate accuracy using EER threshold
    preds = (all_similarities > metrics['eer_threshold']).long()
    val_accuracy = simple_accuracy(preds.numpy(), all_labels.numpy())

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)
    logger.info("EER: %2.5f" % metrics['eer'])
    logger.info("ROC AUC: %2.5f" % metrics['auc'])
    logger.info("EER Threshold: %2.5f" % metrics['eer_threshold'])
    
    writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=global_step)
    writer.add_scalar("test/eer", scalar_value=metrics['eer'], global_step=global_step)
    writer.add_scalar("test/auc", scalar_value=metrics['auc'], global_step=global_step)
        
    return val_accuracy

def evaluate_checkpoint(args, model, test_loader):
    """Load checkpoint and evaluate on test set"""
    logger.info("***** Loading checkpoint and evaluating on test set *****")
    
    # Load checkpoint
    global_step, best_acc = load_checkpoint(args, model)
    
    if global_step == 0:
        logger.warning("No checkpoint found, using current model state")
    
    # Evaluate on test set
    model.eval()
    all_similarities, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            batch = tuple(t.to(args.device) for t in batch)
            img1, img2, labels = batch
            
            # Get embeddings for both images
            emb1 = model(img1, return_embedding=True)
            emb2 = model(img2, return_embedding=True)
            
            # Calculate similarity
            emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
            emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
            cos_sim = torch.sum(emb1 * emb2, dim=1)
            
            if len(all_similarities) == 0:
                all_similarities.append(cos_sim.detach().cpu())
                all_labels.append(labels.detach().cpu())
            else:
                all_similarities[0] = torch.cat([all_similarities[0], cos_sim.detach().cpu()], dim=0)
                all_labels[0] = torch.cat([all_labels[0], labels.detach().cpu()], dim=0)
    
    all_similarities, all_labels = all_similarities[0], all_labels[0]
    
    # Calculate verification metrics
    metrics = calculate_verification_metrics(all_similarities, all_labels)
    
    # Calculate accuracy using EER threshold
    preds = (all_similarities > metrics['eer_threshold']).long()
    test_accuracy = simple_accuracy(preds.numpy(), all_labels.numpy())
    
    logger.info("\n" + "="*50)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("="*50)
    logger.info("Test Accuracy: %2.5f" % test_accuracy)
    logger.info("EER: %2.5f" % metrics['eer'])
    logger.info("ROC AUC: %2.5f" % metrics['auc'])
    logger.info("EER Threshold: %2.5f" % metrics['eer_threshold'])
    logger.info("="*50)
    
    return test_accuracy, metrics

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["NDTWIN","CUB_200_2011"], default="NDTWIN",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/home/mossbee/Work/Dataset')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/home/mossbee/Work/twin_verification/adaptation/TransFG/pretrained/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./output/NDTWIN", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    # Verification-specific parameters
    parser.add_argument("--margin", default=0.4, type=float,
                        help="Margin for contrastive loss in verification task.")
    parser.add_argument("--similarity_threshold", default=0.0, type=float,
                        help="Threshold for similarity-based prediction during training.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    
    # Checkpoint and evaluation arguments
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from checkpoint")
    parser.add_argument('--evaluate', action='store_true',
                        help="Only evaluate checkpoint on test set (no training)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1 if torch.cuda.is_available() else 0
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s, 16-bits training: %s" %
                   (args.device, args.n_gpu, args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    
    # Prepare dataset
    train_loader, test_loader = get_loader(args)
    
    if args.evaluate:
        # Only evaluate checkpoint on test set
        evaluate_checkpoint(args, model, test_loader)
    else:
        # Training
        train_verification(args, model)

if __name__ == "__main__":
    main()