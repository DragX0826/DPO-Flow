# streaming_preprocess_v7.py
import tarfile, os, gc, torch, json, requests, gzip, io, shutil, sys
from tqdm import tqdm
from rdkit import rdBase
from max_flow.data.featurizer import ProteinLigandFeaturizer
from max_flow.utils.pca import compute_pca_rotation, apply_canonicalization

# 1. 強力屏蔽 RDKit 警告，防止日誌堆積導致 OOM
rdBase.DisableLog('rdApp.*')

def stream_v7(archive_url, save_root, max_items=100000):
    os.makedirs(save_root, exist_ok=True)
    featurizer = ProteinLigandFeaturizer()
    
    # 更嚴格的緩存控制
    pdb_cache = {} 
    shard_buffer = []
    
    # 自動檢測現有進度
    existing_shards = [f for f in os.listdir(save_root) if f.startswith("shard_") and f.endswith(".pt")]
    shard_count = len(existing_shards)
    processed_count = shard_count * 500
    
    print(f"🚀 V7 超輕量穩定版啟動 | 目標: {max_items} 對 | 已有進度: {processed_count}")
    
    try:
        response = requests.get(archive_url, stream=True, timeout=120)
        response.raise_for_status()
        
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
            for member in tqdm(tar, desc="🔍 SOTA 採集", mininterval=60.0, disable=False):
                if not member.isfile(): continue
                name = member.name
                
                # 蛋白識別
                if name.endswith("_rec.pdb"):
                    prefix = name.replace("_rec.pdb", "")
                    pdb_cache[prefix] = tar.extractfile(member).read()
                    
                    # 極簡緩存：只留 500 個蛋白
                    if len(pdb_cache) > 500:
                        pdb_cache.pop(next(iter(pdb_cache)))
                    continue
                
                # 配體識別
                if "_rec_" in name and (name.endswith(".sdf.gz") or name.endswith(".sdf")):
                    prefix = name.split("_rec_")[0]
                    if prefix in pdb_cache:
                        try:
                            content = tar.extractfile(member).read()
                            if name.endswith(".gz"):
                                with gzip.GzipFile(fileobj=io.BytesIO(content)) as f_gz:
                                    content = f_gz.read()
                            
                            p_p, l_p = "/tmp/tp.pdb", "/tmp/tl.sdf"
                            with open(p_p, "wb") as f: f.write(pdb_cache[prefix])
                            with open(l_p, "wb") as f: f.write(content)
                            
                            data = featurizer(p_p, l_p)
                            if data is not None:
                                r, c = compute_pca_rotation(data.pos_P)
                                data.pos_P = apply_canonicalization(data.pos_P, torch.zeros(data.pos_P.size(0), dtype=torch.long), r, c)
                                data.pos_L = apply_canonicalization(data.pos_L, torch.zeros(data.pos_L.size(0), dtype=torch.long), r, c)
                                shard_buffer.append(data)
                                processed_count += 1
                                
                                if len(shard_buffer) >= 500:
                                    s_name = f"shard_{shard_count}.pt"
                                    torch.save(shard_buffer, os.path.join(save_root, s_name))
                                    print(f"✅ Shard {shard_count} Saved | Total: {processed_count}")
                                    shard_count += 1
                                    shard_buffer = []
                                    # 每 500 個強制回收一次
                                    gc.collect()
                                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                            # 每 100 個數據點主動清理臨時文件
                            if processed_count % 100 == 0:
                                os.remove(p_p); os.remove(l_p)
                        except: pass

                if processed_count >= max_items: break

    except Exception as e:
        print(f"⚠️ 遇到意外中斷: {e} | 目前產分片數: {shard_count}")

    # 結算
    if shard_buffer:
        torch.save(shard_buffer, os.path.join(save_root, f"shard_{shard_count}.pt"))
    print(f"🎉 V7 採集結束。")

if __name__ == "__main__":
    stream_v7("https://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3.tgz", "/kaggle/working/data")
