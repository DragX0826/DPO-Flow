# streaming_preprocess_v4.py
import tarfile, os, gc, torch, json, argparse, requests, gzip, io
from tqdm import tqdm
from maxflow.data.featurizer import ProteinLigandFeaturizer
from maxflow.utils.pca import compute_pca_rotation, apply_canonicalization

def stream_extract_and_process(archive_url, save_root):
    os.makedirs(save_root, exist_ok=True)
    featurizer = ProteinLigandFeaturizer()
    buffer = {} # Key: directory_prefix, Value: {'pdb': bytes, 'sdf': bytes}
    shard_buffer = []
    shard_count, processed_count = 0, 0
    
    print(f"🚀 V4 強化版啟動 (支持 .gz & _rec.pdb)... 目標: {save_root}")
    response = requests.get(archive_url, stream=True, timeout=60)
    response.raise_for_status()
    
    logged = 0
    with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
        for member in tqdm(tar, desc="🔍 SOTA 掃描"):
            if not member.isfile(): continue
            name = member.name
            
            # --- 核心匹配邏輯 ---
            mode = None
            if name.endswith("_rec.pdb"):
                prefix = name.replace("_rec.pdb", "")
                mode = "pdb"
            elif name.endswith(".sdf") or name.endswith(".sdf.gz"):
                # 兼容多種命名格式
                for s in ["_ligand.sdf.gz", "_sub.sdf.gz", "_docked.sdf.gz", "_min.sdf.gz", "_ligand.sdf", ".sdf"]:
                    if name.endswith(s):
                        prefix = name.replace(s, "")
                        break
                mode = "sdf"
            
            if mode:
                try:
                    content = tar.extractfile(member).read()
                    # 自動解壓 .gz
                    if name.endswith(".gz"):
                        with gzip.GzipFile(fileobj=io.BytesIO(content)) as f_gz:
                            content = f_gz.read()
                    
                    if prefix not in buffer: buffer[prefix] = {}
                    buffer[prefix][mode] = content
                    
                    # 匹配成功！
                    if "pdb" in buffer[prefix] and "sdf" in buffer[prefix]:
                        p_p, l_p = f"/tmp/p_{processed_count}.pdb", f"/tmp/l_{processed_count}.sdf"
                        with open(p_p, "wb") as f: f.write(buffer[prefix]["pdb"])
                        with open(l_p, "wb") as f: f.write(buffer[prefix]["sdf"])
                        
                        data = featurizer(p_p, l_path=l_p)
                        if data is not None:
                            # PCA 歸一化 (Phase 1-10 核心)
                            r, c = compute_pca_rotation(data.pos_P)
                            data.pos_P = apply_canonicalization(data.pos_P, torch.zeros(data.pos_P.size(0), dtype=torch.long), r, c)
                            data.pos_L = apply_canonicalization(data.pos_L, torch.zeros(data.pos_L.size(0), dtype=torch.long), r, c)
                            data.pca_rot, data.pca_center = r, c
                            
                            shard_buffer.append(data)
                            processed_count += 1
                            if len(shard_buffer) >= 500:
                                torch.save(shard_buffer, os.path.join(save_root, f"shard_{shard_count}.pt"))
                                print(f"✅ Saved Shard {shard_count} (Total: {processed_count})")
                                shard_count += 1; shard_buffer = []
                        
                        os.remove(p_p); os.remove(l_p)
                        del buffer[prefix]
                except Exception as e:
                    pass

            # 保持緩存平衡
            if len(buffer) > 10000:
                buffer.pop(next(iter(buffer)))

    # 保存剩餘
    if shard_buffer:
        torch.save(shard_buffer, os.path.join(save_root, f"shard_{shard_count}.pt"))
    print(f"🎉 最終完成！成功處理 {processed_count} 個對。")

if __name__ == "__main__":
    stream_extract_and_process(
        "https://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3.tgz", 
        "/kaggle/working/data"
    )
