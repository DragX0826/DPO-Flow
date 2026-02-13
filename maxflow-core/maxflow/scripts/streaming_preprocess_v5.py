# streaming_preprocess_v5.py
import tarfile, os, gc, torch, json, requests, gzip, io, shutil
from tqdm import tqdm
from maxflow.data.featurizer import ProteinLigandFeaturizer
from maxflow.utils.pca import compute_pca_rotation, apply_canonicalization

def stream_v5(archive_url, save_root, max_items=100000):
    # 1. 自動清理舊數據
    if os.path.exists(save_root):
        print(f"🧹 清理目錄: {save_root}")
        shutil.rmtree(save_root)
    os.makedirs(save_root, exist_ok=True)

    featurizer = ProteinLigandFeaturizer()
    buffer = {} # Key: Rec-Anchor, Value: {'pdb': bytes, 'sdf': bytes}
    shard_buffer = []
    shard_count, processed_count = 0, 0
    shards_manifest = []

    print(f"🚀 V5 啟動 | 目標對數: {max_items} | 支持 .gz & Rec-Anchor 匹配")
    response = requests.get(archive_url, stream=True, timeout=60)
    response.raise_for_status()
    
    with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
        for member in tqdm(tar, desc="🔍 SOTA 掃描"):
            if not member.isfile(): continue
            name = member.name
            
            # --- 精確的 Rec-Anchor 提取 ---
            mode = None
            prefix = None
            
            if name.endswith("_rec.pdb"):
                prefix = name.replace("_rec.pdb", "")
                mode = "pdb"
            elif "_rec_" in name and (name.endswith(".sdf.gz") or name.endswith(".sdf")):
                # 配體命名通常為 {path}/{rec}_rec_{ligand}...sdf.gz
                prefix = name.split("_rec_")[0]
                mode = "sdf"
            
            if mode and prefix:
                try:
                    content = tar.extractfile(member).read()
                    if name.endswith(".gz"):
                        with gzip.GzipFile(fileobj=io.BytesIO(content)) as f_gz:
                            content = f_gz.read()
                    
                    if prefix not in buffer: buffer[prefix] = {}
                    buffer[prefix][mode] = content
                    
                    # 匹配成功！
                    if "pdb" in buffer[prefix] and "sdf" in buffer[prefix]:
                        # 隨機臨時路徑避免衝突
                        p_p, l_p = "/tmp/tp.pdb", "/tmp/tl.sdf"
                        with open(p_p, "wb") as f: f.write(buffer[prefix]["pdb"])
                        with open(l_p, "wb") as f: f.write(buffer[prefix]["sdf"])
                        
                        data = featurizer(p_p, l_path=l_p)
                        if data is not None:
                            r, c = compute_pca_rotation(data.pos_P)
                            data.pos_P = apply_canonicalization(data.pos_P, torch.zeros(data.pos_P.size(0), dtype=torch.long), r, c)
                            data.pos_L = apply_canonicalization(data.pos_L, torch.zeros(data.pos_L.size(0), dtype=torch.long), r, c)
                            
                            shard_buffer.append(data)
                            processed_count += 1
                            
                            if len(shard_buffer) >= 500:
                                s_name = f"shard_{shard_count}.pt"
                                s_path = os.path.join(save_root, s_name)
                                torch.save(shard_buffer, s_path)
                                shards_manifest.append({"file": s_name, "size": len(shard_buffer), "start_idx": processed_count-500, "end_idx": processed_count})
                                print(f"✅ Saved Shard {shard_count} ({processed_count}/{max_items})")
                                shard_count += 1; shard_buffer = []
                        
                        os.remove(p_p); os.remove(l_p)
                        del buffer[prefix]
                except:
                    pass

            # 自動停止邏輯
            if processed_count >= max_items:
                print(f"🎯 已達到目標對數 ({max_items})，正在結算...")
                break

            # 保持大緩存窗口
            if len(buffer) > 30000:
                buffer.pop(next(iter(buffer)))

    # 保存最後的餘項
    if shard_buffer:
        s_name = f"shard_{shard_count}.pt"
        torch.save(shard_buffer, os.path.join(save_root, s_name))
        shards_manifest.append({"file": s_name, "size": len(shard_buffer), "start_idx": processed_count-len(shard_buffer), "end_idx": processed_count})

    # 生成 Manifest
    with open(os.path.join(save_root, "shards_manifest.json"), "w") as f_m:
        json.dump(shards_manifest, f_m, indent=4)
    
    print(f"🎉 結算完成！成功產出 {processed_count} 個對，索引已生成。")

if __name__ == "__main__":
    stream_v5(
        "https://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3.tgz", 
        "/kaggle/working/data",
        max_items=100000
    )
