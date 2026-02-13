import tarfile
import os
import gc
import torch
import json
import argparse
import requests
from tqdm import tqdm
from max_flow.data.featurizer import ProteinLigandFeaturizer
from max_flow.utils.pca import compute_pca_rotation, apply_canonicalization

# ========== Memory-Safe Constants ==========
MAX_BUFFER_SIZE = 5000  # Processing buffer (raw files)
SHARD_SIZE = 2000       # Number of graphs per output shard file (Optimized for Kaggle IO)
GC_INTERVAL = 5000      

class BufferedStream:
    """A simple buffered wrapper for requests stream to speed up tarfile access."""
    def __init__(self, response, chunk_size=1024*1024):
        self.response = response
        self.iterator = response.iter_content(chunk_size=chunk_size)
        self.buffer = b""

    def read(self, size):
        while len(self.buffer) < size:
            try:
                self.buffer += next(self.iterator)
            except StopIteration:
                break
        res, self.buffer = self.buffer[:size], self.buffer[size:]
        return res

def stream_extract_and_process(archive_url, save_root, max_items=None):
    os.makedirs(save_root, exist_ok=True)
    featurizer = ProteinLigandFeaturizer()
    
    # Raw file buffer (pdb/sdf content)
    buffer = {}
    
    # Processed graph buffer (for sharding)
    shard_buffer = []
    shard_count = 0
    
    processed_count = 0
    # We will save a manifest of shards instead of listing all 6M items
    shards_manifest = [] 
    
    iteration_count = 0

    print(f"🚀 Memory-Safe Streaming from: {archive_url}")
    print(f"📦 Shard Size: {SHARD_SIZE} | Buffer Limit: {MAX_BUFFER_SIZE}")
    
    response = requests.get(archive_url, stream=True, timeout=60)
    response.raise_for_status()
    
    stream = BufferedStream(response, chunk_size=1024*1024)
    
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tqdm(tar, desc="SOTA Sharded Stream", mininterval=30.0):
                iteration_count += 1
                
                if iteration_count % GC_INTERVAL == 0:
                    gc.collect()
                
                if not member.isfile(): continue
                
                name = member.name
                if "_pocket10.pdb" in name:
                    prefix, file_type = name.replace("_pocket10.pdb", ""), "pdb"
                elif name.endswith(".sdf"):
                    prefix = name.split("_sub")[0] if "_sub" in name else name.replace(".sdf", "")
                    file_type = "sdf"
                else:
                    continue

                content = tar.extractfile(member).read()
                if prefix not in buffer: buffer[prefix] = {}
                buffer[prefix][file_type] = content

                # Evict from raw buffer if too full
                if len(buffer) > MAX_BUFFER_SIZE:
                    oldest_keys = list(buffer.keys())[:MAX_BUFFER_SIZE // 2]
                    for k in oldest_keys:
                        del buffer[k]
                    gc.collect()

                # Processing Logic
                if "pdb" in buffer[prefix] and "sdf" in buffer[prefix]:
                    try:
                        tmp_pdb = f"/tmp/{prefix.replace('/', '_')}_p.pdb"
                        tmp_sdf = f"/tmp/{prefix.replace('/', '_')}_l.sdf"
                        os.makedirs(os.path.dirname(tmp_pdb) if os.path.dirname(tmp_pdb) else "/tmp", exist_ok=True)
                        
                        with open(tmp_pdb, "wb") as f_pdb: f_pdb.write(buffer[prefix]["pdb"])
                        with open(tmp_sdf, "wb") as f_sdf: f_sdf.write(buffer[prefix]["sdf"])
                        
                        data = featurizer(tmp_pdb, tmp_sdf)
                        
                        if data is not None:
                            # Apply Canonicalization
                            rot, center = compute_pca_rotation(data.pos_P)
                            data.pos_P = apply_canonicalization(data.pos_P, torch.zeros(data.pos_P.size(0), dtype=torch.long), rot, center)
                            data.pos_L = apply_canonicalization(data.pos_L, torch.zeros(data.pos_L.size(0), dtype=torch.long), rot, center)
                            data.pca_rot, data.pca_center = rot, center
                            
                            # Add to Shard Buffer
                            shard_buffer.append(data)
                            processed_count += 1
                            
                            # Flush Shard if full
                            if len(shard_buffer) >= SHARD_SIZE:
                                shard_name = f"shard_{shard_count}.pt"
                                save_path = os.path.join(save_root, shard_name)
                                torch.save(shard_buffer, save_path)
                                
                                shards_manifest.append({
                                    "file": shard_name,
                                    "size": len(shard_buffer),
                                    "start_idx": processed_count - len(shard_buffer),
                                    "end_idx": processed_count
                                })
                                
                                print(f"💾 Saved Shard {shard_count} ({len(shard_buffer)} items) -> {save_path}")
                                shard_buffer = [] # Clear buffer
                                shard_count += 1
                                gc.collect()
                            
                        os.remove(tmp_pdb); os.remove(tmp_sdf)
                    except Exception as e:
                        pass
                    del buffer[prefix]
                
                if max_items and processed_count >= max_items:
                    break
        
        # Save remaining items in buffer
        if len(shard_buffer) > 0:
            shard_name = f"shard_{shard_count}.pt"
            save_path = os.path.join(save_root, shard_name)
            torch.save(shard_buffer, save_path)
            shards_manifest.append({
                "file": shard_name,
                "size": len(shard_buffer),
                "start_idx": processed_count - len(shard_buffer),
                "end_idx": processed_count
            })
            print(f"💾 Saved Final Shard {shard_count} ({len(shard_buffer)} items)")

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"\n❌ Streaming error: {e}")

    # Save Manifest
    with open(os.path.join(save_root, "shards_manifest.json"), "w") as f_man:
        json.dump(shards_manifest, f_man, indent=4)
        
    print(f"\n🎉 Finished! Processed {processed_count} items into {len(shards_manifest)} shards.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="/kaggle/working/data")
    parser.add_argument("--max_items", type=int, default=None)
    args = parser.parse_args()
    stream_extract_and_process(args.archive, args.save_root, args.max_items)
