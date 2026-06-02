import os
import argparse
import wfdb
import numpy as np
import cv2
import pandas as pd
import ast
import matplotlib.pyplot as plt

def load_ptbxl_database(database="ptb-xl/1.0.3"):
    """
    Loads the PTB-XL database metadata CSV to filter by pathology.
    Reads directly from PhysioNet.
    """
    url = f"https://physionet.org/files/{database}/ptbxl_database.csv"
    print("Loading PTB-XL database metadata from PhysioNet...")
    df = pd.read_csv(url, index_col='ecg_id')
    # scp_codes are stored as string dictionaries, convert them
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    return df

def get_records_by_type(df, diagnosis='random', count=5):
    if diagnosis.lower() != 'random':
        # Filter dataframe where the diagnosis is in the scp_codes keys
        filtered = df[df.scp_codes.apply(lambda x: diagnosis in x.keys())]
        if len(filtered) == 0:
            raise ValueError(f"No records found for diagnosis: {diagnosis}")
    else:
        filtered = df
        
    # sample randomly
    if count > len(filtered):
        print(f"Warning: Requested {count} records, but only {len(filtered)} available. Using all.")
        sample = filtered
    else:
        sample = filtered.sample(n=count)
        
    return sample['filename_hr'].tolist()

def fetch_ptbxl_record(record_name, database="ptb-xl/1.0.3"):
    """
    Fetches a real clinical 12-lead ECG from the PTB-XL database on PhysioNet.
    """
    print(f"Fetching record {record_name} ...")
    record_dir, record_file = os.path.split(record_name)
    full_pn_dir = f"{database}/{record_dir}"
    record = wfdb.rdrecord(record_file, pn_dir=full_pn_dir)
    return record

def plot_to_image_custom(record, output_path="temp_ecg.png"):
    """
    Custom 3x4 + 1 rhythm layout using matplotlib for accurate rectangular paper proportions.
    """
    signals = record.p_signal.T # shape: (12, samples)
    fs = record.fs
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig, ax = plt.subplots(figsize=(11.0, 8.0), dpi=150)
    ax.set_xlim(0, 10)
    
    baselines = [7.5, 5.0, 2.5, 0] # y-coordinates for row 1, 2, 3, rhythm
    ax.set_ylim(-1.5, 9.5)
    
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    
    ax.grid(which='major', color='#ff9999', linestyle='-', linewidth=1.2)
    ax.grid(which='minor', color='#ffcccc', linestyle='-', linewidth=0.6)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    
    time = np.arange(signals.shape[1]) / fs
    col_bounds = [(0, 2.5), (2.5, 5.0), (5.0, 7.5), (7.5, 10.0)]
    
    for row in range(3):
        for col in range(4):
            lead_idx = col * 3 + row
            if lead_idx < 12:
                t_start, t_end = col_bounds[col]
                idx_start = int(t_start * fs)
                idx_end = int(t_end * fs)
                
                t_segment = time[idx_start:idx_end]
                sig_segment = signals[lead_idx, idx_start:idx_end]
                
                baseline = baselines[row]
                ax.plot(t_segment, sig_segment + baseline, color='#000033', linewidth=1.0)
                ax.text(t_start + 0.05, baseline + 1.0, leads[lead_idx], fontsize=12, fontweight='bold', color='black', backgroundcolor='white')
                
    rhythm_sig = signals[1, :int(10*fs)]
    ax.plot(time[:int(10*fs)], rhythm_sig + baselines[3], color='#000033', linewidth=1.0)
    ax.text(0.05, baselines[3] + 1.0, 'II', fontsize=12, fontweight='bold', color='black', backgroundcolor='white')
    
    ax.text(8.2, -1.2, '25 mm/s   10 mm/mV', fontsize=10, fontweight='bold', color='black')
    
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)

def apply_realistic_artifacts(image_path, output_path):
    """
    Applies computer vision augmentations to simulate a printed/scanned piece of paper,
    specifically matching the ecglibrary.com classic pinkish style.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
        
    tint = np.array([214, 214, 243], dtype=np.float32) / 255.0
    img = np.clip(img * tint, 0, 255).astype(np.uint8)
    
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    rows, cols, _ = img.shape
    gradient = np.zeros((rows, cols), dtype=np.float32)
    for i in range(cols):
        gradient[:, i] = i / cols
    gradient = np.stack([gradient]*3, axis=-1)
    
    lighting_variation = 0.90 + 0.10 * gradient
    img_lit = np.clip(img_noisy * lighting_variation, 0, 255).astype(np.uint8)
    
    img_final = cv2.GaussianBlur(img_lit, (3, 3), 0)
    cv2.imwrite(output_path, img_final)
    print(f"Realistic ECG saved to {output_path}")

def generate_ecg(record_id, out_name):
    record = fetch_ptbxl_record(record_name=record_id)
    tmp_file = "temp_ecg.png"
    plot_to_image_custom(record, tmp_file)
    apply_realistic_artifacts(tmp_file, out_name)
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic realistic 12-lead ECG images.")
    parser.add_argument("-n", "--count", type=int, default=5, help="Number of ECG images to generate.")
    parser.add_argument("-t", "--type", type=str, default="random", help="Pathology type (e.g., AFIB, NORM, PVC) or 'random'.")
    parser.add_argument("-o", "--output-dir", type=str, default="output_ecgs", help="Directory to save the generated images.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Preparing to generate {args.count} ECGs of type '{args.type}'...")
    
    try:
        df = load_ptbxl_database()
        records_to_process = get_records_by_type(df, diagnosis=args.type, count=args.count)
        
        for i, record_id in enumerate(records_to_process):
            # Extract just the ID number for cleaner filenames
            record_num = record_id.split('/')[-1].replace('_hr', '')
            out_name = os.path.join(args.output_dir, f"ecg_{args.type}_{record_num}.png")
            
            print(f"[{i+1}/{len(records_to_process)}] Processing...")
            generate_ecg(record_id, out_name)
            
        print("Generation complete!")
        
    except Exception as e:
        print(f"Error during generation: {e}")
