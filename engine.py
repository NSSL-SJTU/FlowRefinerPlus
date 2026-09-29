import math
import sys
from typing import Iterable, Optional

import torch
import os

from timm.data import Mixup
from timm.utils import accuracy
from torchvision import datasets, transforms
from collections import Counter
import torch.nn.functional as F
from tqdm import tqdm
import util.misc as misc
import util.lr_sched as lr_sched
import cv2
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def detect_noise_samples(pngs, encoder2, device, dataset_test, labels_list=None,
                         distance_threshold=2, linkage="ward"):
    label_map = dataset_test.class_to_idx

    features = []
    labels = []
    file_paths = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    for i, png in enumerate(tqdm(pngs, desc="Extracting features")):
        img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: unable to read image {png}")
            continue
        if labels_list is None:
            label_name = os.path.basename(os.path.dirname(png))
            if label_name not in label_map:
                print(f"Warning: label '{label_name}' not found in label map")
                continue
            label = label_map[label_name]
        else:
            label = labels_list[i]

        img = transform(img).unsqueeze(0)
        img = img.to(device, non_blocking=True)
        with torch.no_grad():
            try:
                feature = encoder2.forward_encoder(img, mask_ratio=0)[1]
            except Exception:
                feature = encoder2.forward_features(img)
        features.append(feature.cpu().numpy().flatten())
        labels.append(label)
        file_paths.append(png)

    if len(features) == 0:
        return np.array([]), np.array([]), []

    features = np.array(features)
    labels = np.array(labels)

    print(f"Running AgglomerativeClustering with distance_threshold={distance_threshold}, "
          f"linkage={linkage} (auto K) …")
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, linkage=linkage
    )
    cluster_labels = clustering.fit_predict(features)

    cluster_label_map = {}
    # Cluster view: retain labels above the largest adjacent probability gap.
    for cluster in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster)[0]
        cluster_labels_counter = Counter(labels[indices])
        sorted_counts = cluster_labels_counter.most_common()
        if len(sorted_counts) == 1:
            cluster_label_map[cluster] = {sorted_counts[0][0]}
            continue
        probs = [count / (len(indices) + 1e-12) for _, count in sorted_counts]
        gaps = [probs[i] - probs[i + 1] for i in range(len(probs) - 1)]
        boundary = int(np.argmax(gaps))
        cluster_label_map[cluster] = {label for label, _ in sorted_counts[:boundary + 1]}

    # Class view: select the main clusters of each label, then take the union.
    for label in label_map.values():
        indices = np.where(labels == label)[0]
        if len(indices) == 0:
            continue
        sorted_counts = Counter(cluster_labels[indices]).most_common()
        if len(sorted_counts) == 1:
            main_clusters = {sorted_counts[0][0]}
        else:
            probs = [count / (len(indices) + 1e-12) for _, count in sorted_counts]
            gaps = [probs[i] - probs[i + 1] for i in range(len(probs) - 1)]
            boundary = int(np.argmax(gaps))
            main_clusters = {cluster for cluster, _ in sorted_counts[:boundary + 1]}
        for cluster in main_clusters:
            cluster_label_map[cluster].add(label)

    clean_indices = []
    clean_file_paths = []
    clean_labels = []

    for i in range(len(features)):
        if labels[i] in cluster_label_map[cluster_labels[i]]:
            clean_indices.append(i)
            clean_file_paths.append(file_paths[i])
            clean_labels.append(labels[i])

    label_coverage = set(clean_labels)
    for label in label_map.values():
        if label not in label_coverage:
            candidate_indices = np.where(labels == label)[0]
            cluster_counts = Counter(cluster_labels[candidate_indices])
            top_clusters = [cluster for cluster, _ in cluster_counts.most_common(2)]

            for cluster in top_clusters:
                indices = np.where((cluster_labels == cluster) & (labels == label))[0]
                for idx in indices:
                    clean_indices.append(idx)
                    clean_file_paths.append(file_paths[idx])
                    clean_labels.append(labels[idx])
                    label_coverage.add(label)


    clean_labels = np.array(clean_labels)
    clean_file_paths = np.array(clean_file_paths)

    all_indices = set(range(len(labels)))
    clean_indices_set = set(clean_indices)
    noise_indices = list(all_indices - clean_indices_set)
    noise_file_paths = [file_paths[i] for i in noise_indices]

    return clean_file_paths, clean_labels, noise_file_paths



@torch.no_grad()
def label_correction(pngs, encoder1, device, dataset_test, labels_list=None,):
    label_map = dataset_test.class_to_idx
    labels = []
    pseudo_labels = []
    top_probs = []
    file_paths = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    for i, png in enumerate(tqdm(pngs, desc="Extracting features")):
        img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: unable to read image {png}")
            continue
        img = transform(img).unsqueeze(0)
        images = img.to(device, non_blocking=True)

        if labels_list is None:
            label_name = os.path.basename(os.path.dirname(png))
            if label_name not in label_map:
                print(f"Warning: label '{label_name}' not found in label map")
                continue
            label = label_map[label_name]
        else:
            label = int(labels_list[i])
        labels.append(label)

        with torch.no_grad():
            output = encoder1(images)

        probabilities = F.softmax(output, dim=1)
        prob_values, indices = torch.sort(probabilities, descending=True, dim=1)
        file_paths.append(png)
        pseudo_labels.append(int(indices[0][0].item()))
        top_probs.append(prob_values[0][0].item())

    # Keep the small-sample guard used in the journal experiment code.
    if len(file_paths) < 10:
        print("Too few samples for GMM label correction.")
        return [], []

    confidences = np.array(top_probs, dtype=np.float32).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(confidences)
    order = np.argsort(gmm.means_.flatten())
    high_component = int(order[1])
    components = gmm.predict(confidences)

    # High confidence: use the prediction. Low confidence: retain the given label.
    for i in range(len(file_paths)):
        if components[i] != high_component:
            pseudo_labels[i] = int(labels[i])

    return file_paths, pseudo_labels

def train_one_epoch_encoder_flow(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    pred_all = []
    target_all = []

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        _, pred = outputs.topk(1, 1, True, True)
        pred = pred.t()
        pred_all.extend(pred[0].cpu())
        target_all.extend(targets.cpu())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_encoder_packet(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    pred_all = []
    target_all = []


    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples, packet_level=True)
            loss = criterion(outputs, targets)

        _, pred = outputs.topk(1, 1, True, True)
        pred = pred.t()
        pred_all.extend(pred[0].cpu())
        target_all.extend(targets.cpu())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    pred_all = []
    target_all = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

        pred_all.extend(pred[0].cpu())
        target_all.extend(target.cpu())

        acc1, acc2 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item()/100, n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item()/100, n=batch_size)

    macro = precision_recall_fscore_support(target_all, pred_all, average='weighted')
    cm = confusion_matrix(target_all, pred_all)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.4f} Acc@2 {top5.global_avg:.4f} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc2, losses=metric_logger.loss))
    print(
        '* Pre {macro_pre:.4f} Rec {macro_rec:.4f} F1 {macro_f1:.4f}'
        .format(macro_pre=macro[0], macro_rec=macro[1],
                    macro_f1=macro[2]))

    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['macro_pre'] = macro[0]
    test_state['macro_rec'] = macro[1]
    test_state['macro_f1'] = macro[2]
    test_state['cm'] = cm

    return test_state