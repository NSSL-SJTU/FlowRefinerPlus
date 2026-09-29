import os
import glob
import binascii
from PIL import Image
import scapy.all as scapy
from tqdm import tqdm
import numpy as np
import random
import shutil
from collections import defaultdict

def makedir(path):
    try:
        os.mkdir(path)
    except Exception as E:
        pass


def read_MFR_bytes(pcap_dir):
    packets = scapy.rdpcap(pcap_dir)
    data = []
    for packet in packets:
        header = (binascii.hexlify(bytes(packet['IP']))).decode()
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''
        if len(header) > 160:
            header = header[:160]
        elif len(header) < 160:
            header += '0' * (160 - len(header))
        if len(payload) > 480:
            payload = payload[:480]
        elif len(payload) < 480:
            payload += '0' * (480 - len(payload))
        data.append((header, payload))
        if len(data) >= 5:
            break
    if len(data) < 5:
        for i in range(5-len(data)):
            data.append(('0'*160, '0'*480))
    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p
    return final_data

def MFR_generator(flows_pcap_path, output_path):
    flows = glob.glob(flows_pcap_path + "/*/*/*.pcap")
    makedir(output_path)
    makedir(output_path + "/train")
    makedir(output_path + "/test")
    classes = glob.glob(flows_pcap_path + "/*/*")
    for cla in tqdm(classes):
        makedir(cla.replace(flows_pcap_path, output_path))
    for flow in tqdm(flows):
        content = read_MFR_bytes(flow)
        content = np.array([int(content[i:i + 2], 16) for i in range(0, len(content), 2)])
        fh = np.reshape(content, (40, 40))
        fh = np.uint8(fh)
        im = Image.fromarray(fh)
        im.save(flow.replace('.pcap', '.png').replace(flows_pcap_path, output_path))

def make_noise_dataset(old_path, noise_ratio):
    # Construct original dataset paths
    train_path = os.path.join(old_path, 'train', '*', '*')
    test_path = os.path.join(old_path, 'test', '*', '*')

    # Name the new noisy dataset path
    new_path = old_path.split('/')[-1] + f'_noisy{int(100 * noise_ratio)}'
    print(new_path)

    new_train_path = os.path.join(new_path, 'train')
    new_test_path = os.path.join(new_path, 'test')


    train_pngs = glob.glob(train_path)

    labels = [os.path.basename(os.path.dirname(png)) for png in train_pngs]
    unique_labels = list(set(labels))

    label_count = defaultdict(int)
    for label in labels:
        label_count[label] += 1

    for label in unique_labels:
        os.makedirs(os.path.join(new_train_path, label), exist_ok=True)
        os.makedirs(os.path.join(new_test_path, label), exist_ok=True)

    noisy_indices = []
    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        num_noisy_samples = int(label_count[label] * noise_ratio)
        noisy_indices.extend(random.sample(indices, num_noisy_samples))

    # Create a label mapping with injected noise
    label_map = {i: labels[i] for i in range(len(labels))}
    for index in noisy_indices:
        original_label = labels[index]
        random_label = random.choice(unique_labels)
        while random_label == original_label:
            random_label = random.choice(unique_labels)
        label_map[index] = random_label

    for i, png in enumerate(train_pngs):
        new_label = label_map[i]
        filename = f"{os.path.basename(os.path.dirname(png))}.{os.path.basename(png)}"
        new_png_path = os.path.join(new_train_path, new_label, filename)
        shutil.copy(png, new_png_path)

    print(f"Label noise injection completed. {len(noisy_indices)} training samples had their labels changed.")

    # Copy test images to the new path (without noise)
    test_pngs = glob.glob(test_path)
    for png in test_pngs:
        label = os.path.basename(os.path.dirname(png))
        new_png_path = os.path.join(new_test_path, label, os.path.basename(png))
        shutil.copy(png, new_png_path)

    print(f"Test set copied. {len(test_pngs)} test samples were transferred.")
