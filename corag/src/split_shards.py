"""
Utility for splitting data into shards and saving them.
"""
import json
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data file')
    parser.add_argument('--num_shards', type=int, required=True, help='Number of shards')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for shard files')
    args = parser.parse_args()
    
    # Load data
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    total_len = len(data)
    num_shards = args.num_shards
    shard_size = total_len // num_shards
    
    print(f"Total data items: {total_len}")
    print(f"Number of shards: {num_shards}")
    print(f"Shard size: {shard_size}")
    
    # Split and save each shard
    os.makedirs(args.output_dir, exist_ok=True)
    
    for shard_id in range(num_shards):
        start_idx = shard_id * shard_size
        if shard_id == num_shards - 1:
            # Last shard gets remaining items
            end_idx = total_len
        else:
            end_idx = start_idx + shard_size
        
        shard_data = data[start_idx:end_idx]
        shard_path = os.path.join(args.output_dir, f'hotpot_shard_{shard_id}.json')
        
        with open(shard_path, 'w', encoding='utf-8') as f:
            json.dump(shard_data, f, indent=2)
        
        print(f"Shard {shard_id}: items {start_idx} to {end_idx-1} ({len(shard_data)} items) -> {shard_path}")
